package sobek

import (
	"fmt"
	"math"
	"math/big"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/grafana/sobek/parser"
)

// MemoryLimitError is the value of the *InterruptedError that Run*() returns once the allocation budget set with
// SetMemoryLimit is exceeded.
type MemoryLimitError struct{ Limit, Used int64 }

func (e *MemoryLimitError) Error() string {
	return fmt.Sprintf("memory limit exceeded: used %d of %d bytes", e.Used, e.Limit)
}

// StepLimitError is the value of the *InterruptedError that Run*() returns once the deterministic VM instruction /
// native loop step budget set with SetMaxSteps is exceeded.
type StepLimitError struct{ Limit, Used int64 }

func (e *StepLimitError) Error() string {
	return fmt.Sprintf("step limit exceeded: executed %d of %d steps", e.Used, e.Limit)
}

// JobLimitError is the value of the *InterruptedError that Run*() returns once the promise microtask job budget
// set with SetMaxPromiseJobs is exceeded.
type JobLimitError struct{ Limit, Queued int64 }

func (e *JobLimitError) Error() string {
	return fmt.Sprintf("promise job limit exceeded: queued %d of %d jobs", e.Queued, e.Limit)
}

// SetMaxPromiseJobs sets a monotonic limit on total promise microtask jobs enqueued in this runtime (0 = unlimited).
// Exceeding it aborts execution with an uncatchable *InterruptedError whose Value() is a *JobLimitError.
func (r *Runtime) SetMaxPromiseJobs(maxJobs int64) {
	r.jobLimit = maxJobs
}

// PromiseJobCount returns the total number of promise microtask jobs enqueued so far.
func (r *Runtime) PromiseJobCount() int64 {
	return r.jobsQueued
}

// ResetSandboxCounters resets the monotonic memory, step, promise job, and tick counters and clears any interrupt
// flag so a same-tenant Runtime (typically hardened with FreezeIntrinsics) can be reused cleanly across invocations
// in a pool. Never pool a Runtime across untrusted tenant boundaries.
func (r *Runtime) ResetSandboxCounters() {
	r.memUsed = 0
	r.stepsUsed = 0
	r.jobsQueued = 0
	r.ticks = 0
	r.ClearInterrupt()
}

func (r *Runtime) chargePromiseJob() {
	if r.jobsQueued < math.MaxInt64 {
		r.jobsQueued++
	}
	if r.jobLimit > 0 && r.jobsQueued > r.jobLimit {
		ex := &InterruptedError{iface: &JobLimitError{Limit: r.jobLimit, Queued: r.jobsQueued}}
		ex.stack = r.vm.captureStack(nil, 0)
		panic(ex)
	}
}

// CompileSandbox compiles untrusted JavaScript in strict mode with source maps unconditionally disabled (preventing
// //# sourceMappingURL= local filesystem reads / SMB probes / OOM during compilation) and bounded source size and
// parser nesting depth. Defaults: maxSourceBytes = 256 KiB if <= 0, maxNestingDepth = 256 if <= 0.
func CompileSandbox(name, src string, maxSourceBytes, maxNestingDepth int) (*Program, error) {
	if maxSourceBytes <= 0 {
		maxSourceBytes = 256 << 10
	}
	if len(src) > maxSourceBytes {
		return nil, fmt.Errorf("source size %d exceeds sandbox limit of %d bytes", len(src), maxSourceBytes)
	}
	if maxNestingDepth <= 0 {
		maxNestingDepth = 256
	}
	return CompileWithOptions(
		name,
		src,
		true,
		parser.WithDisableSourceMaps,
		parser.WithMaxNestingDepth(maxNestingDepth),
	)
}

// SandboxOptions configures a hardened multi-tenant sandbox profile via Runtime.ApplySandbox.
type SandboxOptions struct {
	MemoryLimitBytes int64
	MaxSteps         int64
	MaxPromiseJobs   int64
	RegExpTimeout    time.Duration
	MaxCallStackSize int
	MaxNestingDepth  int
	DisableCodeGen   bool
	GarbleSafeJSON   bool // sets TagFieldNameMapper("json", false) so garble struct obfuscation cannot break JS names
	FreezeIntrinsics bool // freezes core global prototypes and namespaces against prototype pollution
}

// DefaultSandboxOptions returns a conservative multi-tenant plugin sandbox configuration suitable for untrusted code.
func DefaultSandboxOptions() SandboxOptions {
	return SandboxOptions{
		MemoryLimitBytes: 16 << 20,
		MaxSteps:         2_000_000,
		MaxPromiseJobs:   10_000,
		RegExpTimeout:    150 * time.Millisecond,
		MaxCallStackSize: 256,
		MaxNestingDepth:  256,
		DisableCodeGen:   true,
		GarbleSafeJSON:   true,
		FreezeIntrinsics: false,
	}
}

// ApplySandbox applies all configured sandbox limits, safe parser options (source maps disabled + nesting cap),
// optional garble-safe field name mapping, and optional intrinsic freezing in one call.
func (r *Runtime) ApplySandbox(opts SandboxOptions) {
	if opts.MemoryLimitBytes > 0 {
		r.SetMemoryLimit(opts.MemoryLimitBytes)
	}
	if opts.MaxSteps > 0 {
		r.SetMaxSteps(opts.MaxSteps)
	}
	if opts.MaxPromiseJobs > 0 {
		r.SetMaxPromiseJobs(opts.MaxPromiseJobs)
	}
	if opts.RegExpTimeout > 0 {
		r.SetRegExpTimeout(opts.RegExpTimeout)
	}
	if opts.MaxCallStackSize > 0 {
		r.SetMaxCallStackSize(opts.MaxCallStackSize)
	}
	r.SetCodeGenDisabled(opts.DisableCodeGen)
	depth := opts.MaxNestingDepth
	if depth <= 0 {
		depth = 256
	}
	r.SetParserOptions(parser.WithDisableSourceMaps, parser.WithMaxNestingDepth(depth))
	if opts.GarbleSafeJSON {
		r.SetFieldNameMapper(TagFieldNameMapper("json", false))
	}
	if opts.FreezeIntrinsics {
		r.FreezeIntrinsics()
	}
}

// FreezeIntrinsics freezes core global prototypes and built-in namespaces (Object, Array, Function, String, Number,
// Boolean, RegExp, Error, Promise, Map, Set, Date, JSON, Math, Reflect) so untrusted scripts cannot perform
// prototype pollution. Note that once frozen, scripts cannot overwrite inherited methods like toString on plain
// objects via assignment in strict mode (Object.defineProperty still works).
func (r *Runtime) FreezeIntrinsics() {
	defer r.pauseMem()()
	names := []string{
		"Object", "Array", "Function", "String", "Number", "Boolean",
		"RegExp", "Error", "TypeError", "RangeError", "SyntaxError", "EvalError",
		"Promise", "Map", "Set", "Date", "Symbol", "BigInt",
		"JSON", "Math", "Reflect",
	}
	for _, name := range names {
		v := r.Get(name)
		if obj, ok := v.(*Object); ok {
			if proto := obj.Get("prototype"); proto != nil {
				if pobj, ok := proto.(*Object); ok {
					r.object_freeze(FunctionCall{Arguments: []Value{pobj}})
				}
			}
			r.object_freeze(FunctionCall{Arguments: []Value{obj}})
		}
	}
}

// SetMaxSteps sets a deterministic monotonic CPU step budget (0 = unlimited, the upstream behaviour). Steps are
// accounted every 100 VM instructions and every 4096 native loop iterations, requiring no timers or goroutines.
// Exceeding it aborts execution with an uncatchable *InterruptedError whose Value() is a *StepLimitError.
func (r *Runtime) SetMaxSteps(steps int64) {
	r.stepLimit = steps
}

// StepCount returns the VM/native steps executed so far (monotonic, never decreases).
func (r *Runtime) StepCount() int64 {
	return r.stepsUsed
}

// ExportJSON safely serializes v to UTF-8 JSON inside the runtime (respecting SetMemoryLimit, maxNativeDepth,
// and circular reference detection) and aborts early if the encoded size exceeds maxBytes (> 0). Use this at the
// host boundary instead of unchecked Export() when accepting values from untrusted plugins.
func (r *Runtime) ExportJSON(v Value, maxBytes int64) ([]byte, error) {
	if v == nil || IsUndefined(v) {
		return []byte("null"), nil
	}
	var out []byte
	err := r.runWrapped(func() {
		ctx := _builtinJSON_stringifyContext{
			r:        r,
			allAscii: true,
			maxBytes: maxBytes,
		}
		if ctx.do(v) {
			ctx.charge()
			out = make([]byte, ctx.buf.Len())
			copy(out, ctx.buf.Bytes())
		} else {
			out = []byte("null")
		}
	})
	if err != nil {
		return nil, err
	}
	return out, nil
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

// SetRegExpTimeout bounds one match of any RegExp pattern (all patterns use regexp2); 0 = none. A timeout
// throws a catchable RangeError. It applies to RegExp objects created after the call.
func (r *Runtime) SetRegExpTimeout(d time.Duration) {
	r.regexpTimeout = d
}

// SetCodeGenDisabled disallows dynamic code generation from strings (direct/indirect eval, Function, AsyncFunction,
// GeneratorFunction, including via `(() => {}).constructor`). Calling any of them throws a catchable EvalError.
func (r *Runtime) SetCodeGenDisabled(disabled bool) {
	r.codeGenDisabled = disabled
}

// CodeGenDisabled reports whether dynamic code generation from strings is disallowed on this Runtime.
func (r *Runtime) CodeGenDisabled() bool {
	return r.codeGenDisabled
}

const (
	maxNativeDepth = 10000
	// ponytail: object/function/promise shell sizes are measured constants, not unsafe.Sizeof sums; revisit if the
	// Go object layout changes.
	memObject    = 128
	memProp      = 64
	memFunc      = 256
	memEntry     = 96
	memWeakEntry = 160 // map entry + two runtime.AddCleanup registrations
	memPromise   = 256
	// memIter is the per-iteration garbage (index string, boxing, call args) of a generic loop over a script-controlled
	// length; without it a 1-byte-per-index buffer needs ~64M iterations to reach a 64 MiB budget.
	memIter = 32
	memJob  = 160
)

var (
	valueSize   = int64(unsafe.Sizeof(Value(nil)))
	contextSize = int64(unsafe.Sizeof(context{}))
)

// chargeMem is called BEFORE an allocation of ~n bytes. Built-ins can't be interrupted (Interrupt only fires
// between instructions), so the limit must panic here, inside the native call, like pushCtx does.
func (r *Runtime) chargeMem(n int64) {
	if n <= 0 || r.memPaused > 0 {
		return
	}
	if r.memUsed > math.MaxInt64-n {
		r.memUsed = math.MaxInt64
	} else {
		r.memUsed += n
	}
	if r.memLimit > 0 && r.memUsed > r.memLimit {
		ex := &InterruptedError{iface: &MemoryLimitError{Limit: r.memLimit, Used: r.memUsed}}
		ex.stack = r.vm.captureStack(nil, 0)
		panic(ex)
	}
}

// chargeMul charges count*elemSize with int64 overflow saturation so huge counts never wrap negative and bypass chargeMem.
func (r *Runtime) chargeMul(count, elemSize int64) {
	if count <= 0 || elemSize <= 0 {
		return
	}
	if count > math.MaxInt64/elemSize {
		r.chargeMem(math.MaxInt64)
		return
	}
	r.chargeMem(count * elemSize)
}

func (r *Runtime) chargeValues(n int64) { r.chargeMul(n, valueSize) }

// chargeBigBits charges a BigInt result of ~bits bits before it is computed: `2n**(2n**40n)` or `1n<<10n**12n` is one
// allocation of many GiB, a fatal Go OOM rather than a panic. Add/Sub grow by a bit, so only Mul/Exp/shifts need it.
func (r *Runtime) chargeBigBits(bits float64) {
	if math.IsNaN(bits) || bits <= 0 {
		return
	}
	if bits >= 1<<60 {
		r.chargeMem(math.MaxInt64)
		return
	}
	r.chargeMem(int64(bits/8) + 48)
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

// chargeSteps accounts for n executed VM instructions or native loop ticks.
func (r *Runtime) chargeSteps(n int64) {
	if n <= 0 {
		return
	}
	if r.stepsUsed > math.MaxInt64-n {
		r.stepsUsed = math.MaxInt64
	} else {
		r.stepsUsed += n
	}
	if r.stepLimit > 0 && r.stepsUsed > r.stepLimit {
		ex := &InterruptedError{iface: &StepLimitError{Limit: r.stepLimit, Used: r.stepsUsed}}
		ex.stack = r.vm.captureStack(nil, 0)
		panic(ex)
	}
}

// tick is called once per iteration of a loop whose bound is script-controlled.
func (r *Runtime) tick() {
	r.ticks++
	if r.ticks&4095 == 0 {
		r.chargeSteps(4096)
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

// memCap is the memory held by the saved stacks of a suspended generator/async frame.
func (e *execCtx) memCap() int64 {
	return int64(cap(e.stack))*valueSize + int64(cap(e.refStack))*valueSize +
		int64(cap(e.tryStack))*int64(unsafe.Sizeof(tryFrame{})) + int64(cap(e.iterStack))*int64(unsafe.Sizeof(iterStackItem{}))
}

var sparseItemSize = int64(unsafe.Sizeof(sparseArrayItem{}))
