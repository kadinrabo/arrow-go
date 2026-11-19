// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build go1.18

package exprs

import (
	"fmt"
	"strings"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/compute"
	"github.com/apache/arrow-go/v18/arrow/scalar"
	"github.com/substrait-io/substrait-go/v9/expr"
	"github.com/substrait-io/substrait-go/v9/extensions"
)

const (
	arithmeticFuncsURN = extensions.SubstraitDefaultURNPrefix + "functions_arithmetic"
	comparisonFuncsURN = extensions.SubstraitDefaultURNPrefix + "functions_comparison"
	booleanFuncsURN    = extensions.SubstraitDefaultURNPrefix + "functions_boolean"
	stringFuncsURN     = extensions.SubstraitDefaultURNPrefix + "functions_string"
	icebergSetFuncURN  = "extension:apache.iceberg:functions_set"
)

// toArrowCall maps a Substrait scalar function and its converted arguments
// to the compute expression that runs it.
type toArrowCall = func(sf *expr.ScalarFunction, args []compute.Expression) (compute.Expression, error)

var scalarFunctions = func() map[extensions.FunctionID]toArrowCall {
	m := make(map[extensions.FunctionID]toArrowCall)
	for _, fn := range []string{"add", "subtract", "multiply", "divide", "power", "sqrt", "abs"} {
		m[extensions.FunctionID{URN: arithmeticFuncsURN, Name: fn}] = overflowableArithmetic(fn)
	}
	for _, fn := range []string{"equal", "not_equal", "lt", "lte", "gt", "gte", "is_null", "is_not_null", "is_nan"} {
		m[extensions.FunctionID{URN: comparisonFuncsURN, Name: fn}] = renamedFunction
	}
	m[extensions.FunctionID{URN: booleanFuncsURN, Name: "and"}] = kleene(compute.And, true)
	m[extensions.FunctionID{URN: booleanFuncsURN, Name: "or"}] = kleene(compute.Or, false)
	m[extensions.FunctionID{URN: booleanFuncsURN, Name: "not"}] = renamedFunction
	m[extensions.FunctionID{URN: icebergSetFuncURN, Name: "is_in"}] = setLookup
	m[extensions.FunctionID{URN: stringFuncsURN, Name: "starts_with"}] = startsWith
	return m
}()

// lookupScalarFunction finds the conversion for sf by its simple name, since
// sf.ID() names the variant it resolved to, as in "gt:any_any".
func lookupScalarFunction(sf *expr.ScalarFunction) (toArrowCall, bool) {
	id := sf.ID()
	id.Name, _, _ = strings.Cut(id.Name, ":")
	conv, ok := scalarFunctions[id]
	return conv, ok
}

var arrowFunctionNames = map[string]string{
	"lt":  "less",
	"gt":  "greater",
	"lte": "less_equal",
	"gte": "greater_equal",
}

// arrowName is the Arrow function for sf's simple name.
func arrowName(sf *expr.ScalarFunction) string {
	name, _, _ := strings.Cut(sf.Name(), ":")
	if arrow, ok := arrowFunctionNames[name]; ok {
		return arrow
	}
	return name
}

func renamedFunction(sf *expr.ScalarFunction, args []compute.Expression) (compute.Expression, error) {
	return compute.NewCall(arrowName(sf), args, nil), nil
}

// kleene folds Substrait's variadic and/or into nested calls to Arrow's
// two-argument kleene functions. With no arguments it's the identity.
func kleene(join func(lhs, rhs compute.Expression, ops ...compute.Expression) compute.Expression, identity bool) toArrowCall {
	return func(_ *expr.ScalarFunction, args []compute.Expression) (compute.Expression, error) {
		switch len(args) {
		case 0:
			return compute.NewLiteral(identity), nil
		case 1:
			return args[0], nil
		}
		return join(args[0], args[1], args[2:]...), nil
	}
}

func overflowableArithmetic(name string) toArrowCall {
	return func(sf *expr.ScalarFunction, args []compute.Expression) (compute.Expression, error) {
		overflow, err := parseOption(sf, "overflow", &overflowParser,
			[]overflowBehavior{overflowSILENT, overflowERROR}, overflowSILENT)
		if err != nil {
			return nil, fmt.Errorf("%w: %s overflow %v", err, name, sf.GetOption("overflow"))
		}
		if overflow == overflowSILENT {
			return compute.NewCall(name+"_unchecked", args, nil), nil
		}
		return compute.NewCall(name, args, nil), nil
	}
}

// literalArg returns the value of the literal args[i] of a two-argument
// function, for Arrow functions that take it as an option rather than as an
// argument.
func literalArg(sf *expr.ScalarFunction, args []compute.Expression, i int, what string) (compute.Datum, error) {
	if len(args) != 2 {
		return nil, fmt.Errorf("%w: %s expects 2 arguments, got %d",
			arrow.ErrNotImplemented, sf.Name(), len(args))
	}
	lit, ok := args[i].(*compute.Literal)
	if !ok {
		return nil, fmt.Errorf("%w: %s %s must be a literal", arrow.ErrNotImplemented, sf.Name(), what)
	}
	return lit.Literal, nil
}

func setLookup(sf *expr.ScalarFunction, args []compute.Expression) (compute.Expression, error) {
	values, err := literalArg(sf, args, 1, "value set")
	if err != nil {
		return nil, err
	}

	opts := &compute.SetOptions{NullBehavior: compute.NullMatchingMatch, ValueSet: values}
	if sc, ok := values.(*compute.ScalarDatum); ok {
		list, ok := sc.Value.(*scalar.List)
		if !ok {
			return nil, fmt.Errorf("%w: %s value set of type %s",
				arrow.ErrNotImplemented, sf.Name(), sc.Type())
		}
		opts.ValueSet = compute.NewDatumWithoutOwning(list.Value)
	}
	return compute.NewCall(arrowName(sf), args[:1], opts), nil
}

// startsWith maps CASE_INSENSITIVE to IgnoreCase. A case_sensitivity
// preference list with only CASE_INSENSITIVE_ASCII returns an error
// wrapping arrow.ErrNotImplemented.
func startsWith(sf *expr.ScalarFunction, args []compute.Expression) (compute.Expression, error) {
	sensitivity, err := parseOption(sf, "case_sensitivity", &caseSensitivityParser,
		[]caseSensitivity{caseSensitive, caseInsensitive}, caseSensitive)
	if err != nil {
		return nil, fmt.Errorf("%w: starts_with case_sensitivity %v", err, sf.GetOption("case_sensitivity"))
	}

	pattern, err := literalArg(sf, args, 1, "pattern")
	if err != nil {
		return nil, err
	}
	sd, ok := pattern.(*compute.ScalarDatum)
	if !ok {
		return nil, fmt.Errorf("%w: starts_with pattern must be a scalar", arrow.ErrNotImplemented)
	}
	sc := sd.Value
	if !sc.IsValid() {
		return nil, fmt.Errorf("%w: starts_with pattern must not be null", arrow.ErrNotImplemented)
	}
	if ext, ok := sc.(*scalar.Extension); ok {
		sc = ext.Value
	}
	str, ok := sc.(scalar.BinaryScalar)
	if !ok {
		return nil, fmt.Errorf("%w: starts_with pattern of type %s", arrow.ErrNotImplemented, sc.DataType())
	}

	return compute.NewCall("starts_with", args[:1], &compute.MatchSubstringOptions{
		Pattern:    string(str.Data()),
		IgnoreCase: sensitivity == caseInsensitive,
	}), nil
}

type overflowBehavior string

const (
	overflowSILENT   overflowBehavior = "SILENT"
	overflowSATURATE overflowBehavior = "SATURATE"
	overflowERROR    overflowBehavior = "ERROR"
)

type enumParser[typ ~string] struct {
	values map[typ]struct{}
}

func (e *enumParser[typ]) parse(v string) (typ, error) {
	out := typ(v)
	if _, ok := e.values[out]; ok {
		return out, nil
	}
	return "", arrow.ErrNotFound
}

var overflowParser = enumParser[overflowBehavior]{
	values: map[overflowBehavior]struct{}{
		overflowSILENT:   {},
		overflowSATURATE: {},
		overflowERROR:    {},
	},
}

type caseSensitivity string

const (
	caseSensitive        caseSensitivity = "CASE_SENSITIVE"
	caseInsensitive      caseSensitivity = "CASE_INSENSITIVE"
	caseInsensitiveASCII caseSensitivity = "CASE_INSENSITIVE_ASCII"
)

var caseSensitivityParser = enumParser[caseSensitivity]{
	values: map[caseSensitivity]struct{}{
		caseSensitive:        {},
		caseInsensitive:      {},
		caseInsensitiveASCII: {},
	},
}

// parseOption returns the first preference for optionName that is one of
// implemented, or def if sf doesn't set the option.
func parseOption[typ ~string](sf *expr.ScalarFunction, optionName string, parser *enumParser[typ], implemented []typ, def typ) (typ, error) {
	opts := sf.GetOption(optionName)
	if len(opts) == 0 {
		return def, nil
	}

	for _, o := range opts {
		p, err := parser.parse(o)
		if err != nil {
			return def, arrow.ErrInvalid
		}
		for _, i := range implemented {
			if i == p {
				return p, nil
			}
		}
	}

	return def, arrow.ErrNotImplemented
}
