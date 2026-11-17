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
	"unsafe"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/compute"
	"github.com/apache/arrow-go/v18/arrow/decimal128"
	"github.com/apache/arrow-go/v18/arrow/endian"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/arrow/scalar"
	"github.com/substrait-io/substrait-go/v9/expr"
	"github.com/substrait-io/substrait-go/v9/types"
)

func intervalYearToMonthDatum(mem memory.Allocator, years, months int32) (compute.Datum, error) {
	bldr := array.NewInt32Builder(mem)
	defer bldr.Release()

	bldr.Append(years)
	bldr.Append(months)
	arr := bldr.NewArray()
	defer arr.Release()
	return &compute.ScalarDatum{Value: scalar.NewExtensionScalar(
		scalar.NewFixedSizeListScalar(arr), intervalYear())}, nil
}

func literalToDatum(mem memory.Allocator, lit expr.Literal, r typeResolver) (compute.Datum, error) {
	switch v := lit.(type) {
	case *expr.PrimitiveLiteral[bool]:
		return compute.NewDatum(scalar.NewBooleanScalar(v.Value)), nil
	case *expr.PrimitiveLiteral[int8]:
		return compute.NewDatum(scalar.NewInt8Scalar(v.Value)), nil
	case *expr.PrimitiveLiteral[int16]:
		return compute.NewDatum(scalar.NewInt16Scalar(v.Value)), nil
	case *expr.PrimitiveLiteral[int32]:
		return compute.NewDatum(scalar.NewInt32Scalar(v.Value)), nil
	case *expr.PrimitiveLiteral[int64]:
		return compute.NewDatum(scalar.NewInt64Scalar(v.Value)), nil
	case *expr.PrimitiveLiteral[float32]:
		return compute.NewDatum(scalar.NewFloat32Scalar(v.Value)), nil
	case *expr.PrimitiveLiteral[float64]:
		return compute.NewDatum(scalar.NewFloat64Scalar(v.Value)), nil
	case *expr.PrimitiveLiteral[string]:
		return compute.NewDatum(scalar.NewStringScalar(v.Value)), nil
	case *expr.PrimitiveLiteral[types.Timestamp]:
		return compute.NewDatum(scalar.NewTimestampScalar(arrow.Timestamp(v.Value), &arrow.TimestampType{Unit: arrow.Microsecond})), nil
	case *expr.PrimitiveLiteral[types.TimestampTz]:
		return compute.NewDatum(scalar.NewTimestampScalar(arrow.Timestamp(v.Value),
			&arrow.TimestampType{Unit: arrow.Microsecond, TimeZone: timestampTzTimezone})), nil
	case *expr.PrimitiveLiteral[types.Date]:
		return compute.NewDatum(scalar.NewDate32Scalar(arrow.Date32(v.Value))), nil
	case *expr.PrimitiveLiteral[types.Time]:
		return compute.NewDatum(scalar.NewTime64Scalar(arrow.Time64(v.Value), &arrow.Time64Type{Unit: arrow.Microsecond})), nil
	case *expr.PrimitiveLiteral[types.FixedChar]:
		length := int(v.Type.(*types.FixedCharType).Length)
		return compute.NewDatum(scalar.NewExtensionScalar(
			scalar.NewFixedSizeBinaryScalar(memory.NewBufferBytes([]byte(v.Value)),
				&arrow.FixedSizeBinaryType{ByteWidth: length}), fixedChar(int32(length)))), nil
	case *expr.ByteSliceLiteral[[]byte]:
		return compute.NewDatum(scalar.NewBinaryScalar(memory.NewBufferBytes(v.Value), arrow.BinaryTypes.Binary)), nil
	case *expr.ByteSliceLiteral[types.UUID]:
		return compute.NewDatum(scalar.NewExtensionScalar(scalar.NewFixedSizeBinaryScalar(
			memory.NewBufferBytes(v.Value), uuid().(arrow.ExtensionType).StorageType()), uuid())), nil
	case *expr.ByteSliceLiteral[types.FixedBinary]:
		return compute.NewDatum(scalar.NewFixedSizeBinaryScalar(memory.NewBufferBytes(v.Value),
			&arrow.FixedSizeBinaryType{ByteWidth: int(v.Type.(*types.FixedBinaryType).Length)})), nil
	case *expr.NullLiteral:
		dt, _, err := fromSubstraitType(v.Type, r)
		if err != nil {
			return nil, err
		}
		return &compute.ScalarDatum{Value: scalar.MakeNullScalar(dt)}, nil
	case *expr.ListLiteral:
		var elemType arrow.DataType

		values := make([]scalar.Scalar, len(v.Value))
		for i, val := range v.Value {
			d, err := literalToDatum(mem, val, r)
			if err != nil {
				return nil, err
			}
			defer d.Release()
			values[i] = d.(*compute.ScalarDatum).Value
			if elemType != nil {
				if !arrow.TypeEqual(values[i].DataType(), elemType) {
					return nil, fmt.Errorf("%w: %s has a value whose type doesn't match the other list values",
						arrow.ErrInvalid, v)
				}
			} else {
				elemType = values[i].DataType()
			}
		}

		bldr := array.NewBuilder(memory.DefaultAllocator, elemType)
		defer bldr.Release()
		if err := scalar.AppendSlice(bldr, values); err != nil {
			return nil, err
		}
		arr := bldr.NewArray()
		defer arr.Release()
		return compute.NewDatum(scalar.NewListScalar(arr)), nil
	case *expr.MapLiteral:
		dt, _, err := fromSubstraitType(v.Type, r)
		if err != nil {
			return nil, err
		}

		mapType, ok := dt.(*arrow.MapType)
		if !ok {
			return nil, fmt.Errorf("%w: map literal with non-map type", arrow.ErrInvalid)
		}

		keys, values := make([]scalar.Scalar, len(v.Value)), make([]scalar.Scalar, len(v.Value))
		for i, kv := range v.Value {
			k, err := literalToDatum(mem, kv.Key, r)
			if err != nil {
				return nil, err
			}
			defer k.Release()
			scalarKey := k.(*compute.ScalarDatum).Value

			v, err := literalToDatum(mem, kv.Value, r)
			if err != nil {
				return nil, err
			}
			defer v.Release()
			scalarValue := v.(*compute.ScalarDatum).Value

			if !arrow.TypeEqual(mapType.KeyType(), scalarKey.DataType()) {
				return nil, fmt.Errorf("%w: key type mismatch for %s, got key with type %s",
					arrow.ErrInvalid, mapType, scalarKey.DataType())
			}
			if !arrow.TypeEqual(mapType.ItemType(), scalarValue.DataType()) {
				return nil, fmt.Errorf("%w: value type mismatch for %s, got value with type %s",
					arrow.ErrInvalid, mapType, scalarValue.DataType())
			}

			keys[i], values[i] = scalarKey, scalarValue
		}

		keyBldr, valBldr := array.NewBuilder(mem, mapType.KeyType()), array.NewBuilder(mem, mapType.ItemType())
		defer keyBldr.Release()
		defer valBldr.Release()

		if err := scalar.AppendSlice(keyBldr, keys); err != nil {
			return nil, err
		}
		if err := scalar.AppendSlice(valBldr, values); err != nil {
			return nil, err
		}

		keyArr, valArr := keyBldr.NewArray(), valBldr.NewArray()
		defer keyArr.Release()
		defer valArr.Release()

		kvArr, err := array.NewStructArray([]arrow.Array{keyArr, valArr}, []string{"key", "value"})
		if err != nil {
			return nil, err
		}
		defer kvArr.Release()

		return &compute.ScalarDatum{Value: scalar.NewMapScalar(kvArr)}, nil
	case *expr.StructLiteral:
		fields := make([]scalar.Scalar, len(v.Value))
		names := make([]string, len(v.Value))

		for i, l := range v.Value {
			lit, err := literalToDatum(mem, l, r)
			if err != nil {
				return nil, err
			}
			fields[i] = lit.(*compute.ScalarDatum).Value
		}

		s, err := scalar.NewStructScalarWithNames(fields, names)
		return compute.NewDatum(s), err
	case expr.IntervalYearToMonthLiteral:
		return intervalYearToMonthDatum(mem, v.Years, v.Months)
	case *expr.IntervalYearToMonthLiteral:
		return intervalYearToMonthDatum(mem, v.Years, v.Months)
	case *expr.ProtoLiteral:
		switch t := v.Type.(type) {
		case *types.DecimalType:
			byts := v.Value.([]byte)
			if len(byts) != arrow.Decimal128SizeBytes {
				return nil, fmt.Errorf("%w: decimal literal had %d bytes (expected %d)",
					arrow.ErrInvalid, len(byts), arrow.Decimal128SizeBytes)
			}

			var val decimal128.Num
			data := (*(*[arrow.Decimal128SizeBytes]byte)(unsafe.Pointer(&val)))[:]
			copy(data, byts)
			if endian.IsBigEndian {
				// reverse the bytes
				for i := len(data)/2 - 1; i >= 0; i-- {
					opp := len(data) - 1 - i
					data[i], data[opp] = data[opp], data[i]
				}
			}

			return compute.NewDatum(scalar.NewDecimal128Scalar(val,
				&arrow.Decimal128Type{Precision: t.Precision, Scale: t.Scale})), nil
		case *types.UserDefinedType: // not yet implemented
		case *types.IntervalYearToMonthType:
			val := v.Value.(*types.IntervalYearToMonth)
			return intervalYearToMonthDatum(mem, val.Years, val.Months)
		case *types.IntervalDayType:
			bldr := array.NewInt32Builder(mem)
			defer bldr.Release()

			val := v.Value.(*types.IntervalDayToSecond)
			typ := intervalDay()
			bldr.Append(val.Days)
			bldr.Append(val.Seconds)
			arr := bldr.NewArray()
			defer arr.Release()
			return &compute.ScalarDatum{Value: scalar.NewExtensionScalar(
				scalar.NewFixedSizeListScalar(arr), typ)}, nil
		case *types.VarCharType:
			return compute.NewDatum(scalar.NewExtensionScalar(
				scalar.NewStringScalar(v.Value.(string)), varChar(int32(t.Length)))), nil
		case *types.PrecisionTimeType:
			dt, _, err := fromSubstraitType(t, r)
			if err != nil {
				return nil, err
			}

			switch dt.ID() {
			case arrow.TIME32:
				return &compute.ScalarDatum{Value: scalar.NewTime32Scalar(
					arrow.Time32(v.Value.(int64)), dt)}, nil
			case arrow.TIME64:
				return &compute.ScalarDatum{Value: scalar.NewTime64Scalar(
					arrow.Time64(v.Value.(int64)), dt)}, nil
			}
		case *types.PrecisionTimestampType:
			dt, _, err := fromSubstraitType(t, r)
			if err != nil {
				return nil, err
			}

			return &compute.ScalarDatum{Value: scalar.NewTimestampScalar(
				arrow.Timestamp(v.Value.(int64)), dt)}, nil
		case *types.PrecisionTimestampTzType:
			dt, _, err := fromSubstraitType(t, r)
			if err != nil {
				return nil, err
			}

			return &compute.ScalarDatum{Value: scalar.NewTimestampScalar(
				arrow.Timestamp(v.Value.(int64)), dt)}, nil
		}
	}

	return nil, arrow.ErrNotImplemented
}
