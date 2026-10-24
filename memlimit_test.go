package sobek

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

const mib = 1 << 20

// Budgets: every memory test below runs at 2 MiB (4 MiB where noted), over-budget allocations are 2-5x that, and loops
// that the budget is meant to stop carry an explicit cap, so a broken limiter costs a few MB instead of the machine.
// The only 64 MiB run is TestMemoryLimitProductionBudget. Assertions use the VM's own counter (MemoryUsage) and the
// interrupt error, never heap or process statistics; ReadMemStats appears only in the TestMain guard.
const budget = 2 * mib

// currentTest is the last test that entered one of the helpers below, for the heap guard's report.
var currentTest atomic.Value

func enter(t *testing.T) { currentTest.Store(t.Name()) }

// TestMain is a hard guard for a broken limiter: GOMEMLIMIT is soft, so a poller exits the whole run when the Go heap
// passes 2 GiB (a runaway BigInt/make can commit 100+ GB before anything else notices).
func TestMain(m *testing.M) {
	go func() {
		var ms runtime.MemStats
		for {
			time.Sleep(50 * time.Millisecond)
			runtime.ReadMemStats(&ms)
			if ms.HeapAlloc > 1<<31 {
				name, _ := currentTest.Load().(string)
				fmt.Fprintf(os.Stderr, "HEAP GUARD: HeapAlloc %d MiB during %q: the memory limiter let an allocation through\n", ms.HeapAlloc>>20, name)
				os.Exit(3)
			}
		}
	}()
	os.Exit(m.Run())
}

// slack scales wall-clock bounds for the race detector (5-10x slower).
func slack(d time.Duration) time.Duration {
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, s := range bi.Settings {
			if s.Key == "-race" && s.Value == "true" {
				return d * 8
			}
		}
	}
	return d
}

func isMemLimit(err error) bool {
	var ie *InterruptedError
	var me *MemoryLimitError
	return errors.As(err, &ie) && errors.As(err, &me)
}

// runLimited runs src under the given budget and requires the memory-limit error within d.
func runLimited(t *testing.T, src string, limit int64, d time.Duration) *Runtime {
	t.Helper()
	enter(t)
	vm := New()
	vm.SetMemoryLimit(limit)
	start := time.Now()
	_, err := vm.RunString(src)
	took := time.Since(start)
	var me *MemoryLimitError
	if !isMemLimit(err) || !errors.As(err, &me) {
		t.Fatalf("%q: want *InterruptedError{*MemoryLimitError}, got %T %v (usage %d of %d)", src, err, err, vm.MemoryUsage(), limit)
	}
	if me.Limit != limit || me.Used <= limit || vm.MemoryUsage() < me.Used { // unwinding may still charge a few objects
		t.Fatalf("%q: inconsistent counters: error %+v, MemoryUsage %d", src, me, vm.MemoryUsage())
	}
	if took > slack(d) {
		t.Fatalf("%q: took %v, want < %v", src, took, slack(d))
	}
	if vm.memPaused != 0 || vm.nativeDepth != 0 {
		t.Fatalf("%q: leaked memPaused=%d nativeDepth=%d", src, vm.memPaused, vm.nativeDepth)
	}
	return vm
}

const loop256k = `for(let i=0;i<40;i++)s+="x".repeat(1<<18)` // 10 MiB if nothing stops it

func TestMemoryLimitStringGrowth(t *testing.T) {
	runLimited(t, `let s="";`+loop256k, budget, 2*time.Second)
}

func TestMemoryLimitPadEndPrealloc(t *testing.T) {
	runLimited(t, `"x".padEnd(1e7)`, budget, 2*time.Second)
}

func TestMemoryLimitJoinAmplify(t *testing.T) {
	runLimited(t, `new Array(1e5).join("x".repeat(100))`, budget, 2*time.Second)
}

func TestMemoryLimitArrayGrowth(t *testing.T) {
	runLimited(t, `let a=[];for(let i=0;i<60;i++)a.push(new Array(1e4).fill(1))`, budget, 2*time.Second)
}

func TestMemoryLimitNotCatchable(t *testing.T) {
	runLimited(t, `try{let s="";`+loop256k+`}catch(e){"caught"}`, budget, 2*time.Second)
}

func TestMemoryLimitZeroUnlimited(t *testing.T) {
	enter(t)
	vm := New()
	v, err := vm.RunString(`"x".repeat(1<<22).length`)
	if err != nil || v.ToInteger() != 1<<22 {
		t.Fatalf("got %v, %v", v, err)
	}
	if vm.MemoryUsage() < 1<<22 {
		t.Fatalf("usage is tracked even without a limit, got %d", vm.MemoryUsage())
	}
}

// s += x costs the growth only, so building a string piecewise is O(total), not O(n^2).
func TestMemoryLimitConcatNoFalsePositive(t *testing.T) {
	enter(t)
	vm := New()
	vm.SetMemoryLimit(4 * mib)
	v, err := vm.RunString(`let s="";for(let i=0;i<1311;i++)s+="a".repeat(100);s.length`)
	if err != nil || v.ToInteger() != 131100 {
		t.Fatalf("got %v, %v", v, err)
	}
	if u := vm.MemoryUsage(); u >= mib { // quadratic charging would be ~84 MB
		t.Fatalf("usage %d", u)
	}
}

func TestMemoryLimitJSON(t *testing.T) {
	runLimited(t, `JSON.stringify(new Array(1e5).fill("x".repeat(100)))`, budget, 2*time.Second)
}

func TestMemoryLimitMapGrowth(t *testing.T) {
	runLimited(t, `const m=new Map();for(let i=0;i<2e5;i++)m.set(i,i)`, budget, 2*time.Second)
}

func TestMemoryLimitToUpperRetained(t *testing.T) {
	runLimited(t, `const s="x".repeat(1<<16);const a=[];for(let i=0;i<150;i++)a.push(s.toUpperCase())`, budget, 2*time.Second)
}

// Every make() sized from a script-controlled length is charged first. Sizes are 2-5x the budget (16 B per slot).
func TestMemoryLimitHugeMake(t *testing.T) {
	const rxexec = `const re=/a/;re.exec=function(){if(this.d)return null;this.d=1;return {length:5e5,0:"a",index:0}};`
	for name, src := range map[string]string{
		"sort":      `Array.prototype.sort.call({length:5e5})`,
		"apply":     `Math.max.apply(null,{length:5e5})`,
		"replacer":  `JSON.stringify(1,new Proxy([],{get:(t,k)=>k==="length"?5e5:t[k]}))`,
		"from":      `Array.from({length:5e5})`,
		"tosorted":  `Array.prototype.toSorted.call({length:5e5})`,
		"rxrep":     rxexec + `"a".replace(re,"b")`,
		"typed":     `new Uint8Array(8e6)`,
		"arraybuf":  `new ArrayBuffer(8e6)`,
		"repeat":    `"abc".repeat(3e6)`,
		"padStart":  `"".padStart(4e6,"é")`,
		"normalize": `"é".repeat(6e5).normalize("NFD")`, // the 1.2 MB input fits the budget, the normalize charge does not
	} {
		t.Run(name, func(t *testing.T) {
			runLimited(t, `try{`+src+`}catch(e){"caught"}`, budget, 2*time.Second)
		})
	}
}

// Generic loops that grow a Go buffer per iteration over a script-controlled length, finite but past the budget.
func TestMemoryLimitGenericBuffers(t *testing.T) {
	for name, src := range map[string]string{
		"toLocaleString": `Array.prototype.toLocaleString.call({length:5e5})`,
		"raw":            `String.raw({raw:{length:1e6}})`,
		"json":           `JSON.stringify(new Proxy([],{get:(t,k)=>k==="length"?5e5:undefined}))`,
		"split": `class R extends RegExp{exec(s){if(this.d)return null;this.d=1;this.lastIndex=1;return {length:1e6,0:"a",index:0}}};
			"ab".split(new R("a"))`,
		"join": `Array.prototype.join.call({length:100},"x".repeat(1<<16))`,
	} {
		t.Run(name, func(t *testing.T) {
			runLimited(t, src, budget, 2*time.Second)
		})
	}
}

// fleetRow is a fleet-shaped agent report of ~size bytes: strings, numbers, nested rows.
func fleetRow(size int) string {
	var b strings.Builder
	b.WriteString(`{"id":"64f1c0ffee00112233445566","hostname":"WS-0042","os":"Windows 11 Pro","online":true,"cpu":{"name":"Intel(R) Core(TM) i7","cores":8,"load":[1.5,2.5,3.5,4.5]},"software":[`)
	for i := 0; b.Len() < size; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, `{"name":"Package number %d","version":"10.2.%d.4","publisher":"Vendor %d Ltd","size":%d,"installed":true}`, i, i, i%17, i*1024)
	}
	b.WriteString(`]}`)
	return b.String()
}

// JSON.parse is charged ~6.2x its input (measured: the real retained heap of this shape is ~7.6x). 16 rows of 25 KB
// fit a 4 MiB budget, 16 rows of 100 KB do not.
func TestMemoryLimitJSONParseFleet(t *testing.T) {
	enter(t)
	const src = `const rows=[];for(let i=0;i<16;i++)rows.push(JSON.parse(ROW));rows.length`
	for _, tc := range []struct {
		size  int
		limit bool
	}{{25 << 10, false}, {100 << 10, true}} {
		vm := New()
		vm.SetMemoryLimit(4 * mib)
		vm.Set("ROW", fleetRow(tc.size))
		v, err := vm.RunString(src)
		switch {
		case tc.limit && !isMemLimit(err):
			t.Fatalf("%d KB rows: want limit error, got %v (usage %d)", tc.size>>10, err, vm.MemoryUsage())
		case !tc.limit && (err != nil || v.ToInteger() != 16):
			t.Fatalf("%d KB rows: %v %v (usage %d)", tc.size>>10, v, err, vm.MemoryUsage())
		case !tc.limit && vm.MemoryUsage() >= 4*mib:
			t.Fatalf("usage %d", vm.MemoryUsage())
		}
	}
}

// The one run at the production budget (CODE_MODE's 64 MiB): 200 inner results of 25 KB fit.
func TestMemoryLimitProductionBudget(t *testing.T) {
	enter(t)
	vm := New()
	vm.SetMemoryLimit(64 * mib)
	vm.Set("ROW", fleetRow(25<<10))
	v, err := vm.RunString(`const rows=[];for(let i=0;i<200;i++)rows.push(JSON.parse(ROW));rows.length`)
	if err != nil || v.ToInteger() != 200 || vm.MemoryUsage() >= 64*mib {
		t.Fatalf("%v %v usage %d", v, err, vm.MemoryUsage())
	}
}

// Two Runtimes in one process each count only their own allocations: A is interrupted at its budget while B, in the
// middle of its run, allocates about 1 MB and succeeds with a counter that holds none of A's bytes.
func TestMemoryLimitPerVM(t *testing.T) {
	enter(t)
	const limit = 2e6
	bStarted, aDone := make(chan struct{}), make(chan struct{})
	b := New()
	b.SetMemoryLimit(limit)
	b.Set("hold", func() { close(bStarted); <-aDone }) // B is mid-run until A has overrun
	var bv Value
	var berr error
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		bv, berr = b.RunString(`const s="x".repeat(5e5);hold();const u="y".repeat(5e5);s.length+u.length`)
	}()
	<-bStarted
	a := New()
	a.SetMemoryLimit(limit)
	_, aerr := a.RunString(`"x".repeat(4e6)`)
	close(aDone)
	wg.Wait()
	var me *MemoryLimitError
	if !errors.As(aerr, &me) || a.MemoryUsage() <= limit {
		t.Fatalf("A must be interrupted over its budget: %v (usage %d)", aerr, a.MemoryUsage())
	}
	if berr != nil || bv.ToInteger() != 1e6 {
		t.Fatalf("B must succeed: %v %v", bv, berr)
	}
	if u := b.MemoryUsage(); u < 1e6 || u > 1.2e6 || u >= a.MemoryUsage() {
		t.Fatalf("B's counter must hold only B's ~1 MB, got %d (A: %d)", u, a.MemoryUsage())
	}
}

func TestMemoryLimitAsyncChain(t *testing.T) {
	runLimited(t, `let n=0;async function f(){if(++n>5000)return;await null;return f()}f()`, budget, time.Second)
	runLimited(t, `let n=0;function g(){if(++n>5000)return;return Promise.resolve().then(g)}g()`, budget, time.Second)
}

// The per-object constants are measured, not derived. A throwaway heap probe found ~2.0 KB of real retained heap per
// level of these chains; the budget must stay >= half of that or the chain tests stop proving anything.
func TestMemoryLimitChainPerLevel(t *testing.T) {
	enter(t)
	const levels = 2000
	for name, src := range map[string]string{
		"async": `let n=0;async function f(){if(++n==N){probe();return}await null;return f()}f()`,
		"then":  `let n=0;function g(){if(++n==N){probe();return}return Promise.resolve().then(g)}g()`,
	} {
		t.Run(name, func(t *testing.T) {
			vm := New()
			var charged int64
			vm.Set("N", levels)
			vm.Set("probe", func() { charged = vm.MemoryUsage() })
			if _, err := vm.RunString(src); err != nil || charged == 0 {
				t.Fatalf("charged=%d err=%v", charged, err)
			}
			if perLevel := charged / levels; perLevel < 1020 {
				t.Fatalf("charged %d B per level, under half of the measured ~2039 B of real heap", perLevel)
			}
		})
	}
}

// Every kind of retained object charges at least a shell, so a loop of them cannot slip under the budget.
func TestMemoryLimitShellsCharged(t *testing.T) {
	enter(t)
	for name, expr := range map[string]string{
		"map": `new Map()`, "set": `new Set()`, "date": `new Date()`, "error": `new Error("x")`, "regexp": `/x/`,
		"typedArray": `new Uint8Array(1)`, "weakMap": `new WeakMap()`, "emptyArray": `[]`, "stringObject": `new String("x")`,
		"closure": `(function(){return i})`, "class": `class{}`, "promise": `Promise.resolve()`, "object": `{a:1,b:2}`,
		"generator": `(function*(){})()`, "bound": `(function(){}).bind(null)`, "proxy": `new Proxy({},{})`,
	} {
		t.Run(name, func(t *testing.T) {
			vm := New()
			if _, err := vm.RunString(`const a=[];for(let i=0;i<1000;i++)a.push(` + expr + `)`); err != nil {
				t.Fatal(err)
			}
			if perObject := vm.MemoryUsage() / 1000; perObject < memObject {
				t.Fatalf("%s charges %d B per object, under the %d B shell", name, perObject, memObject)
			}
		})
	}
}

// BigInt Mul/Exp/shifts are single huge allocations (a fatal Go OOM), so they are charged before they run.
func TestMemoryLimitBigInt(t *testing.T) {
	for name, src := range map[string]string{
		"exp":     `const b=2n,e=20000000n;b**e`,
		"lsh":     `const e=20000000n;1n<<e`,
		"sar":     `const e=-20000000n;1n>>e`,
		"mul":     `const s=8000000n;const x=1n<<s;x*x`,
		"asUintN": `BigInt.asUintN(2e7,5n)`,
		"asIntN":  `BigInt.asIntN(2e7,5n)`,
	} {
		t.Run(name, func(t *testing.T) {
			runLimited(t, `try{`+src+`}catch(e){"caught"}`, budget, 2*time.Second)
		})
	}
	enter(t)
	v, err := New().RunString(`[1n**(10n**12n), 0n**(10n**12n), (-1n)**(10n**12n+1n), 2n**100n, 3n*4n, 1n<<64n].join()`)
	if err != nil || v.String() != "1,0,-1,1267650600228229401496703205376,12,18446744073709551616" {
		t.Fatalf("small results must not be charged out: %v %v", v, err)
	}
}

// The compiler folds constant expressions in a private VM with no script budget: a huge BigInt literal expression must
// not allocate while compiling (it throws a RangeError when it runs, like V8), and a folded key that falls back to run
// time is charged like any other operation.
func TestCompileTimeConstantBigInt(t *testing.T) {
	enter(t)
	start := time.Now()
	vm := New()
	vm.SetMemoryLimit(budget)
	v, err := vm.RunString(`try{1n<<(10n**12n)}catch(e){e instanceof RangeError}`)
	if err != nil || v.Export() != true {
		t.Fatalf("a huge constant must throw a RangeError when it runs: %v %v", v, err)
	}
	if _, err := vm.RunString(`const k={[2n**(2n**40n)]:1}`); !isMemLimit(err) {
		t.Fatalf("a huge constant property key must hit the budget at run time, got %v", err)
	}
	if v, err := New().RunString(`(1n<<64n)+2n**10n*3n`); err != nil || v.String() != "18446744073709554688" {
		t.Fatalf("a small constant still folds: %v %v", v, err)
	}
	if d := time.Since(start); d > slack(time.Second) {
		t.Fatalf("took %v", d)
	}
}

// Iterating a generator into an array, a spread or a destructuring rest retains one slot per element.
func TestMemoryLimitGeneratorSpread(t *testing.T) {
	const gen = `(function*(){for(let i=0;i<1e6;i++)yield i})()` // 16 MB if nothing stops it
	for name, src := range map[string]string{
		"array": `[...` + gen + `]`,
		"from":  `Array.from(` + gen + `)`,
		"rest":  `const [a,...r]=` + gen,
		"set":   `new Set(` + gen + `)`,
	} {
		t.Run(name, func(t *testing.T) {
			runLimited(t, src, budget, 2*time.Second)
		})
	}
}

// The await re-wrap is exempt from the budget, so a loop that retains nothing ends at the watchdog, not the limit.
func TestAwaitLoopUncharged(t *testing.T) {
	enter(t)
	vm := New()
	vm.SetMemoryLimit(budget)
	time.AfterFunc(200*time.Millisecond, func() { vm.Interrupt("watchdog") })
	_, err := vm.RunString(`async function f(){for(;;){await null}} f()`)
	var ie *InterruptedError
	if !errors.As(err, &ie) || ie.Value() != "watchdog" || isMemLimit(err) {
		t.Fatalf("want the watchdog interrupt, got %v", err)
	}
	if u := vm.MemoryUsage(); u >= 64<<10 {
		t.Fatalf("an await loop retains nothing, charged %d bytes", u)
	}
}

// The exemption never covers user code: a `then` getter on an awaited value is charged.
func TestAwaitThenableGetterCharged(t *testing.T) {
	runLimited(t, `async function f(){await {get then(){let s="";`+loop256k+`}}} f()`, budget, 2*time.Second)
}

// A throwing `constructor` getter on an awaited promise rejects the await (first and later awaits), never an internal
// error, and never leaks the memory pause.
func TestAwaitConstructorGetterThrows(t *testing.T) {
	const mkp = `const p=Promise.resolve(1);Object.defineProperty(p,"constructor",{get(){throw new Error("ctor boom")}});`
	for _, later := range []bool{false, true} {
		pre := ""
		if later {
			pre = "await null;"
		}
		t.Run(fmt.Sprintf("caught/later=%v", later), func(t *testing.T) {
			runLimited(t, mkp+`async function f(){`+pre+`try{await p}catch(e){}; "x".repeat(1e7)} f()`, budget, 2*time.Second)
		})
		t.Run(fmt.Sprintf("uncaught/later=%v", later), func(t *testing.T) {
			enter(t)
			vm := New()
			vm.SetMemoryLimit(budget)
			_, err := vm.RunString(mkp + `let msg; async function f(){` + pre + `await p} f().catch(e=>{msg=e.message})`)
			if err != nil {
				t.Fatalf("RunString must return normally, got %v", err)
			}
			if got := vm.Get("msg"); got == nil || got.String() != "ctor boom" {
				t.Fatalf("f() should reject with the getter's error, got %v", got)
			}
			if vm.memPaused != 0 {
				t.Fatalf("leaked pause %d", vm.memPaused)
			}
		})
	}
}

// Interrupt must land inside long natives, not only between instructions.
func TestInterruptIndexOfGeneric(t *testing.T) {
	interruptsWithin(t, `Array.prototype.indexOf.call({length:2**53-1},1)`)
}

func TestInterruptJoinGeneric(t *testing.T) {
	interruptsWithin(t, `Array.prototype.join.call({length:2**53-1})`)
}

func interruptsWithin(t *testing.T, src string) {
	t.Helper()
	enter(t)
	vm := New()
	vm.SetMemoryLimit(16 * mib) // backstop only: a missed interrupt would end at this budget, not at OOM
	time.AfterFunc(200*time.Millisecond, func() { vm.Interrupt("stop") })
	start := time.Now()
	_, err := vm.RunString(src)
	var ie *InterruptedError
	if !errors.As(err, &ie) || ie.Value() != "stop" {
		t.Fatalf("%q: want InterruptedError, got %v", src, err)
	}
	if d := time.Since(start); d > slack(time.Second) {
		t.Fatalf("%q: interrupt took %v", src, d)
	}
}

// Native recursion past 10 000 levels throws a catchable RangeError instead of a fatal Go stack overflow.
func nativeDepthRangeError(t *testing.T, src string) {
	t.Helper()
	enter(t)
	vm := New()
	start := time.Now()
	v, err := vm.RunString(`try{` + src + `;"no error"}catch(e){e instanceof RangeError}`)
	if err != nil || v.Export() != true {
		t.Fatalf("want a caught RangeError, got %v %v", v, err)
	}
	if d := time.Since(start); d > slack(time.Second) {
		t.Fatalf("took %v", d)
	}
	if vm.nativeDepth != 0 {
		t.Fatalf("leaked nativeDepth %d", vm.nativeDepth)
	}
}

const deepArray = `let o=[];for(let i=0;i<3e4;i++)o=[o];`

func TestNativeDepthJSONStringify(t *testing.T) {
	nativeDepthRangeError(t, deepArray+`JSON.stringify(o)`)
	v, err := New().RunString(`let o=[];for(let i=0;i<9000;i++)o=[o];JSON.stringify(o).length`)
	if err != nil || v.ToInteger() != 18002 {
		t.Fatalf("a 9000-deep JSON.stringify must still work: %v %v", v, err)
	}
}

func TestNativeDepthJSONParse(t *testing.T) {
	nativeDepthRangeError(t, `JSON.parse("[".repeat(3e4)+"]".repeat(3e4))`)
}

func TestNativeDepthToString(t *testing.T) {
	nativeDepthRangeError(t, deepArray+`String(o)`)
}

func TestNativeDepthFlat(t *testing.T) {
	nativeDepthRangeError(t, deepArray+`o.flat(Infinity)`)
}

func TestRegExpTimeout(t *testing.T) {
	enter(t)
	const redos = `"a".repeat(30)+"b"`
	for name, src := range map[string]string{
		"literal": `/^(a+)+\1$/.test(` + redos + `)`,
		"new":     `new RegExp("^(a+)+\\1$").test(` + redos + `)`,
		"compile": `const r=/x/;r.compile("^(a+)+\\1$");r.test(` + redos + `)`,
	} {
		t.Run(name, func(t *testing.T) {
			vm := New()
			vm.SetRegExpTimeout(200 * time.Millisecond)
			start := time.Now()
			v, err := vm.RunString(`try{` + src + `;"no error"}catch(e){e instanceof RangeError}`)
			if err != nil || v.Export() != true {
				t.Fatalf("want a caught RangeError, got %v %v", v, err)
			}
			if d := time.Since(start); d > slack(2*time.Second) {
				t.Fatalf("took %v", d)
			}
		})
	}
	v, err := New().RunString(`/^(a+)+\1$/.test("aa")`)
	if err != nil || v.Export() != true {
		t.Fatalf("without the setter a short input still matches: %v %v", v, err)
	}
}

// A RE2-served pattern falls back to a lazily compiled regexp2 when matching from a non-zero lastIndex; no timeout
// (the default) must not turn into an immediate one there, and a set timeout must reach it.
func TestRegExpLazyRegexp2Timeout(t *testing.T) {
	enter(t)
	const lazy = `const r=/(a+)+$/g;r.lastIndex=1;r.test("a".repeat(40)+"b")`
	v, err := New().RunString(`const r=/a/g;r.lastIndex=1;r.test("aa")`)
	if err != nil || v.Export() != true {
		t.Fatalf("no timeout set: %v %v", v, err)
	}
	vm := New()
	vm.SetRegExpTimeout(200 * time.Millisecond)
	start := time.Now()
	v, err = vm.RunString(`try{` + lazy + `;"no error"}catch(e){e instanceof RangeError}`)
	if err != nil || v.Export() != true {
		t.Fatalf("want a caught RangeError, got %v %v", v, err)
	}
	if d := time.Since(start); d > slack(2*time.Second) {
		t.Fatalf("took %v", d)
	}
}

// A literal is compiled once per Program and its *regexp2.Regexp is shared by every runtime running it: the timeout
// must be applied to a copy. Run with -race.
func TestRegExpTimeoutSharedLiteralRace(t *testing.T) {
	enter(t)
	prg := MustCompile("lit.js", `try{/^(a+)+\1$/.test(input);"no error"}catch(e){e instanceof RangeError}`, false)
	var wg sync.WaitGroup
	run := func(timeout time.Duration, input string, want interface{}) {
		defer wg.Done()
		vm := New()
		vm.SetRegExpTimeout(timeout)
		vm.Set("input", input)
		v, err := vm.RunProgram(prg)
		if err != nil || v.Export() != want {
			t.Errorf("timeout=%v: got %v %v, want %v", timeout, v, err, want)
		}
	}
	wg.Add(2)
	go run(200*time.Millisecond, strings.Repeat("a", 30)+"b", true)
	go run(0, "ab", "no error")
	wg.Wait()
}

// Array results are charged once (before the allocation), not again when stored on the array object. slice/splice
// pass on their slow path (the species array is `new Array(n)`, never a std fast-path target); map/with/to* are the
// fast paths that doubled.
func TestMemoryChargeSliceOnce(t *testing.T) {
	const n = 10000
	for _, op := range []string{`a.slice()`, `a.splice(0, 1e4)`, `a.map(x => x)`, `a.with(0, 1)`, `a.toReversed()`, `a.toSorted()`, `a.toSpliced(0, 0)`} {
		enter(t)
		vm := New()
		if _, err := vm.RunString(`var a = []; for (var i = 0; i < 1e4; i++) a.push(0)`); err != nil {
			t.Fatal(err)
		}
		before := vm.MemoryUsage()
		if _, err := vm.RunString(op); err != nil {
			t.Fatalf("%s: %v", op, err)
		}
		if d, want := vm.MemoryUsage()-before, n*valueSize; d < want || d > want*3/2 {
			t.Errorf("%s charged %d bytes, want about %d (one charge)", op, d, want)
		}
	}
}
