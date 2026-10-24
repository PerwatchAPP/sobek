package sobek

import (
	"fmt"
	"math"
	"math/big"
	"sync/atomic"
	"time"
	"unsafe"
)

// MemoryLimitError is the value of the *InterruptedError that Run*() returns once the allocation budget set with
// SetMemoryLimit is exceeded.
type MemoryLimitError struct{ Limit, Used int64 }

func (e *MemoryLimitError) Error() string {
	return fmt.Sprintf("memory limit exceeded: used %d of %d bytes", e.Used, e.Limit)
}

// SetMemoryLimit sets a monotonic allocation budget in bytes (0 = unlimited, the upstream behaviour). Allocations are
// charged before they happen and never credited back on GC, so the budget is an upper bound of what a script
// allocated, not of what it retains. Going over it aborts the run with an uncatchable *InterruptedError whose Value()
// is a *MemoryLimitError. Embedder calls outside Run*() (ToValue, NewObject...) are charged too and may panic once the
// budget is spent.
func (r *Runtime) SetMemoryLimit(bytes int64) {
	r.memLimit = bytes
}

// MemoryUsage returns the bytes charged so far (monotonic, never decreases).
func (r *Runtime) MemoryUsage() int64 {
	return r.memUsed
}

// SetRegExpTimeout bounds one match of a regexp2-backed pattern (backreferences, lookbehind...); 0 = none. A timeout
// throws a catchable RangeError. Patterns served by Go's linear RE2 engine need no bound. It applies to RegExp objects
// created after the call.
func (r *Runtime) SetRegExpTimeout(d time.Duration) {
	r.regexpTimeout = d
}

const (
	maxNativeDepth = 10000
	// ponytail: object/function/promise shell sizes are measured constants, not unsafe.Sizeof sums; revisit if the
	// Go object layout changes.
	memObject  = 128
	memProp    = 64
	memFunc    = 256
	memEntry   = 96
	memPromise = 256
	// memIter is the per-iteration garbage (index string, boxing, call args) of a generic loop over a script-controlled
	// length; without it a 1-byte-per-index buffer needs ~64M iterations to reach a 64 MiB budget.
	memIter = 32
	memJob  = 160
)

var valueSize = int64(unsafe.Sizeof(Value(nil)))

// chargeMem is called BEFORE an allocation of ~n bytes. Built-ins can't be interrupted (Interrupt only fires
// between instructions), so the limit must panic here, inside the native call, like pushCtx does.
func (r *Runtime) chargeMem(n int64) {
	if n <= 0 || r.memPaused > 0 {
		return
	}
	r.memUsed += n
	if r.memUsed < 0 {
		r.memUsed = math.MaxInt64
	}
	if r.memLimit > 0 && r.memUsed > r.memLimit {
		ex := &InterruptedError{iface: &MemoryLimitError{Limit: r.memLimit, Used: r.memUsed}}
		ex.stack = r.vm.captureStack(nil, 0)
		panic(ex)
	}
}

func (r *Runtime) chargeValues(n int64) { r.chargeMem(n * valueSize) }

// chargeBigBits charges a BigInt result of ~bits bits before it is computed: `2n**(2n**40n)` or `1n<<10n**12n` is one
// allocation of many GiB, a fatal Go OOM rather than a panic. Add/Sub grow by a bit, so only Mul/Exp/shifts need it.
func (r *Runtime) chargeBigBits(bits float64) {
	r.chargeMem(int64(min(bits, 1<<60)/8) + 48)
}

func (r *Runtime) chargeBigExp(x, y *big.Int) {
	if x.BitLen() > 1 { // 0 and +-1 stay tiny whatever the exponent
		ey, _ := new(big.Float).SetInt(y).Float64()
		r.chargeBigBits(float64(x.BitLen()) * ey)
	}
}

// pauseMem turns chargeMem off until the returned func runs (always `defer r.pauseMem()()`). Only for internal
// garbage that no user code can reach or run inside.
func (r *Runtime) pauseMem() func() {
	r.memPaused++
	return func() { r.memPaused-- }
}

// strMemSize is the cost of a string: UTF-16 for non-ASCII, one byte per char otherwise.
func strMemSize(s String) int64 {
	switch v := s.(type) {
	case asciiString:
		return int64(len(v))
	case unicodeString:
		return 2 * int64(len(v))
	}
	return 2 * int64(s.Length())
}

// memSize is what a StringBuilder holds right now.
func (b *StringBuilder) memSize() int64 {
	return int64(b.asciiBuilder.Len()) + 2*int64(len(b.unicodeBuilder.buf))
}

// chargeBuilder charges the growth of b since *charged and records it.
func (r *Runtime) chargeBuilder(b *StringBuilder, charged *int64) {
	n := b.memSize()
	r.chargeMem(n - *charged)
	*charged = n
}

// checkInterrupt makes Interrupt() land inside long natives, which vm.run's between-instruction poll can't reach.
func (r *Runtime) checkInterrupt() {
	vm := r.vm
	if atomic.LoadUint32(&vm.interrupted) != 0 {
		vm.interruptLock.Lock()
		ex := &InterruptedError{iface: vm.interruptVal}
		ex.stack = vm.captureStack(nil, 0)
		vm.interruptLock.Unlock()
		panic(ex)
	}
}

// tick is called once per iteration of a loop whose bound is script-controlled.
func (r *Runtime) tick() {
	r.ticks++
	if r.ticks&4095 == 0 {
		r.checkInterrupt()
	}
}

// enterNative/leaveNative bracket one level of native recursion; a Go stack overflow is fatal, not a panic.
func (r *Runtime) enterNative() {
	if r.nativeDepth >= maxNativeDepth {
		panic(rangeError("Maximum nesting depth exceeded"))
	}
	r.checkInterrupt()
	r.nativeDepth++
}

func (r *Runtime) leaveNative() { r.nativeDepth-- }

// chargeConcat charges only the growth of a concatenation (total - the larger part), so an `s += x` loop costs
// O(total), not O(n^2). ponytail: a mixed ASCII/UTF-16 pair widens the ASCII side, under-counted by <= 2x.
func (r *Runtime) chargeConcat(a, b String) {
	r.chargeMem(min(strMemSize(a), strMemSize(b)))
}

// memCap is the memory held by the saved stacks of a suspended generator/async frame.
func (e *execCtx) memCap() int64 {
	return int64(cap(e.stack))*valueSize + int64(cap(e.refStack))*valueSize +
		int64(cap(e.tryStack))*int64(unsafe.Sizeof(tryFrame{})) + int64(cap(e.iterStack))*int64(unsafe.Sizeof(iterStackItem{}))
}

var sparseItemSize = int64(unsafe.Sizeof(sparseArrayItem{}))
