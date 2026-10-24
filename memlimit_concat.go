package sobek

import (
	"strings"
	"unsafe"

	"github.com/grafana/sobek/unistring"
)

// Monitic: string concatenation is charged for what it really allocates, and `s += x` builder loops stay cheap because
// they really are cheap: the result is written into spare capacity behind the left operand.
//
// Go strings and unicodeStrings are immutable copies, so `c+n` allocates len(c)+len(n) even when only n is new; charging
// just the growth let 200 MB of `a.push(c+n)` through a 16 MB budget. Every concat is therefore charged its full result
// length, except the in-place append below, which is charged only the bytes it writes.
//
// SAFETY of the in-place append (the only unsafe code in the fork): a Runtime owns a few append buffers. len(buf) is the
// length of the longest string ever handed out from it, and every such string is a prefix buf[:k], k <= len(buf), that
// shares buf's first byte. A concat whose left operand is exactly buf[:len(buf)] (same data pointer, same length: the
// "tail") writes ONLY at indices >= len(buf), which no existing string can see, then grows len(buf). Any other left
// operand (an older prefix, a fork, a foreign string) is copied in full, so a string's bytes are never mutated and a
// second append from the same prefix cannot clobber the first. Capacity-limited (3-index) slices are handed out for
// unicodeString so nothing else can append into the spare room, and the buffer pins its own allocation, so the
// pointer comparison cannot hit a recycled address. Buffers are per Runtime (not goroutine-safe, like the Runtime).
const (
	// concatMinBuf (chars): results shorter than this are plain copies; keeps tiny `a+b` chains from claiming buffers.
	concatMinBuf = 1 << 10
	// concatSlots: live append buffers per kind, least recently used evicted. ponytail: 9+ interleaved large builders
	// thrash the slots and fall back to copy-and-charge-in-full (honest, quadratic); raise if that ever matters.
	concatSlots = 8
)

type concatBufs struct {
	a [][]byte   // ASCII buffers, most recently used first
	u [][]uint16 // UTF-16 buffers (index 0 is the BOM, like unicodeString)
	// Address and length of the last large copying concat result (not a pointer: pins nothing). A later concat whose left
	// operand is exactly that string is a builder loop and gets a buffer; one-off results (`c+n`) never pay the 2x spare.
	lastP uintptr
	lastN int
}

func flatStr(s String) bool {
	switch s.(type) {
	case asciiString, unicodeString:
		return true
	}
	return false
}

func strKey(v Value) (p uintptr, n int) {
	switch s := v.(type) {
	case asciiString:
		return uintptr(unsafe.Pointer(unsafe.StringData(string(s)))), len(s)
	case unicodeString:
		return uintptr(unsafe.Pointer(unsafe.SliceData(s))), len(s)
	}
	return 0, 0
}

func pushFront[T any](s [][]T, b []T) [][]T {
	if len(s) < concatSlots {
		s = append(s, nil)
	}
	copy(s[1:], s) // a full list drops its last (least recently used) buffer
	s[0] = b
	return s
}

func moveFront[T any](s [][]T, i int) {
	b := s[i]
	copy(s[1:i+1], s[:i])
	s[0] = b
}

func (c *concatBufs) findA(s asciiString) bool {
	p := unsafe.StringData(string(s))
	for i, b := range c.a {
		if len(b) == len(s) && unsafe.SliceData(b) == p {
			moveFront(c.a, i)
			return true
		}
	}
	return false
}

func (c *concatBufs) findU(s unicodeString) bool {
	p := unsafe.SliceData(s)
	for i, b := range c.u {
		if len(b) == len(s) && unsafe.SliceData(b) == p {
			moveFront(c.u, i)
			return true
		}
	}
	return false
}

func fillA(b []byte, parts []Value, off int) {
	for _, p := range parts {
		off += copy(b[off:], string(p.(asciiString)))
	}
}

func fillU(b []uint16, parts []Value, off int) {
	for _, p := range parts {
		switch p := p.(type) {
		case asciiString:
			for i := 0; i < len(p); i++ {
				b[off] = uint16(p[i])
				off++
			}
		case unicodeString:
			off += copy(b[off:], p[1:])
		}
	}
}

// concat2 is `a + b` on two JS strings.
func (r *Runtime) concat2(a, b String) String {
	if x, ok := a.(asciiString); ok {
		if y, ok := b.(asciiString); ok && len(x)+len(y) < concatMinBuf {
			r.chargeMem(int64(len(x) + len(y)))
			return x + y
		}
	}
	if flatStr(a) && flatStr(b) {
		parts := [2]Value{a, b}
		return r.concatStrs(parts[:])
	}
	r.chargeMem(strMemSize(a) + strMemSize(b)) // lazily imported strings: plain copy, charged in full
	return a.Concat(b)
}

// concatStrs concatenates parts, each an asciiString or unicodeString, and charges the memory it allocates.
func (r *Runtime) concatStrs(parts []Value) String {
	n, uni := 0, false
	for _, p := range parts {
		switch p := p.(type) {
		case asciiString:
			n += len(p)
		case unicodeString:
			n += len(p) - 1
			uni = true
		}
	}
	toStringLen(n)
	c := &r.cbuf

	// In place: the left operand is the tail of an append buffer.
	switch f := parts[0].(type) {
	case asciiString:
		if !uni && len(f) >= concatMinBuf && c.findA(f) {
			b := c.a[0]
			if cap(b)-len(b) >= n-len(f) {
				r.chargeMem(int64(n - len(f)))
				b = b[:n]
			} else { // full: double, and charge the copy
				r.chargeMem(int64(n))
				nb := make([]byte, n, 2*n)
				copy(nb, f)
				b = nb
			}
			fillA(b, parts[1:], len(f))
			c.a[0] = b
			return asciiString(unsafe.String(unsafe.SliceData(b), n))
		}
	case unicodeString:
		if len(f)-1 >= concatMinBuf && c.findU(f) {
			b := c.u[0]
			if cap(b)-len(b) >= n+1-len(f) {
				r.chargeMem(int64(n+1-len(f)) * 2)
				b = b[:n+1]
			} else {
				r.chargeMem(int64(n) * 2)
				nb := make([]uint16, n+1, 2*(n+1))
				copy(nb, f)
				b = nb
			}
			fillU(b, parts[1:], len(f))
			c.u[0] = b
			return unicodeString(b[: n+1 : n+1])
		}
	}

	// Copy: charged in full. A result that is the next concat's left operand is a builder loop, so it gets a buffer.
	width := int64(1)
	if uni {
		width = 2
	}
	r.chargeMem(int64(n) * width)
	p, pn := strKey(parts[0])
	build := n >= concatMinBuf && pn > 0 && p == c.lastP && pn == c.lastN
	var res String
	switch {
	case !uni && build:
		b := make([]byte, n, 2*n)
		fillA(b, parts, 0)
		c.a = pushFront(c.a, b)
		return asciiString(unsafe.String(unsafe.SliceData(b), n))
	case uni && build:
		b := make([]uint16, n+1, 2*(n+1))
		b[0] = unistring.BOM
		fillU(b, parts, 1)
		c.u = pushFront(c.u, b)
		return unicodeString(b[: n+1 : n+1])
	case !uni:
		var sb strings.Builder
		sb.Grow(n)
		for _, p := range parts {
			sb.WriteString(string(p.(asciiString)))
		}
		res = asciiString(sb.String())
	default:
		var ub unicodeStringBuilder
		ub.Grow(n)
		for _, p := range parts {
			ub.writeString(p.(String))
		}
		res = ub.String()
	}
	if n >= concatMinBuf {
		c.lastP, c.lastN = strKey(res)
	}
	return res
}
