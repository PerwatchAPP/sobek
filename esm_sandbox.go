package sobek

import (
	"fmt"
	"sort"

	"github.com/grafana/sobek/parser"
)

// virtualModuleRecord implements ModuleRecord for an in-memory host-provided ES module (e.g. "@monitic/sdk")
// with zero filesystem access.
type virtualModuleRecord struct {
	name    string
	exports map[string]any
	names   []string
}

var _ ModuleRecord = (*virtualModuleRecord)(nil)

func newVirtualModuleRecord(name string, exports map[string]any) *virtualModuleRecord {
	names := make([]string, 0, len(exports))
	for k := range exports {
		names = append(names, k)
	}
	sort.Strings(names)
	return &virtualModuleRecord{
		name:    name,
		exports: exports,
		names:   names,
	}
}

func (v *virtualModuleRecord) Link() error { return nil }

func (v *virtualModuleRecord) ResolveExport(exportName string, _ ...ResolveSetElement) (*ResolvedBinding, bool) {
	if _, ok := v.exports[exportName]; ok {
		return &ResolvedBinding{
			BindingName: exportName,
			Module:      v,
		}, false
	}
	return nil, false
}

func (v *virtualModuleRecord) GetExportedNames(callback func([]string), _ ...ModuleRecord) bool {
	callback(v.names)
	return true
}

func (v *virtualModuleRecord) Evaluate(rt *Runtime) *Promise {
	p, resolve, _ := rt.NewPromise()
	inst := &virtualModuleInstance{rt: rt, exports: make(map[string]Value, len(v.exports))}
	for k, val := range v.exports {
		if sv, ok := val.(Value); ok {
			inst.exports[k] = sv
		} else {
			inst.exports[k] = rt.ToValue(val)
		}
	}
	_ = resolve(inst)
	return p
}

type virtualModuleInstance struct {
	rt      *Runtime
	exports map[string]Value
}

func (vi *virtualModuleInstance) GetBindingValue(exportName string) Value {
	return vi.exports[exportName]
}

// EvaluateVirtualModule parses, links, and evaluates an ES Module source string inside a closed virtual module
// sandbox. Only specifiers present in virtualModules can be imported (any other specifier fails with an error,
// never touching the filesystem). Source maps are unconditionally disabled and AST nesting depth is capped.
// Returns the module's exported namespace object so the host can invoke exported functions (e.g. `render`, `onAlert`).
func (r *Runtime) EvaluateVirtualModule(name, src string, virtualModules map[string]map[string]any, maxSourceBytes, maxNestingDepth int) (ns *Object, err error) {
	if maxSourceBytes <= 0 {
		maxSourceBytes = 256 << 10
	}
	if len(src) > maxSourceBytes {
		return nil, fmt.Errorf("module source size %d exceeds sandbox limit of %d bytes", len(src), maxSourceBytes)
	}
	if maxNestingDepth <= 0 {
		maxNestingDepth = 256
	}

	records := make(map[string]*virtualModuleRecord, len(virtualModules))
	for spec, exports := range virtualModules {
		records[spec] = newVirtualModuleRecord(spec, exports)
	}

	resolver := func(_ interface{}, specifier string) (ModuleRecord, error) {
		if rec, ok := records[specifier]; ok {
			return rec, nil
		}
		return nil, fmt.Errorf("virtual module %q is not allowed in sandbox", specifier)
	}

	mod, err := ParseModule(name, src, resolver, parser.WithDisableSourceMaps, parser.WithMaxNestingDepth(maxNestingDepth))
	if err != nil {
		return nil, err
	}
	if err := mod.Link(); err != nil {
		return nil, err
	}

	var p *Promise
	err = r.runWrapped(func() {
		p = r.CyclicModuleRecordEvaluate(mod, resolver)
	})
	if err != nil {
		return nil, err
	}
	if p.State() == PromiseStateRejected {
		res := p.Result()
		if ex, ok := res.Export().(error); ok {
			return nil, ex
		}
		return nil, &Exception{val: res}
	}
	if p.State() != PromiseStateFulfilled {
		return nil, fmt.Errorf("virtual module %q did not settle synchronously (top-level await requires an event loop)", name)
	}

	// Collect exported bindings into a plain JS object for ergonomic host access.
	err = r.runWrapped(func() {
		ns = r.NewObject()
		inst := r.modules[mod]
		if inst == nil {
			return
		}
		mod.GetExportedNames(func(names []string) {
			for _, exportName := range names {
				if rb, _ := mod.ResolveExport(exportName); rb != nil {
					if target := r.modules[rb.Module]; target != nil {
						if v := target.GetBindingValue(rb.BindingName); v != nil {
							_ = ns.Set(exportName, v)
						}
					}
				}
			}
		})
	})
	if err != nil {
		return nil, err
	}
	return ns, nil
}
