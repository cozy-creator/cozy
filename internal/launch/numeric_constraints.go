package launch

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
)

func positiveMultipleOf(value json.Number) (*big.Rat, error) {
	approximate, err := value.Float64()
	if err != nil || math.IsNaN(approximate) || math.IsInf(approximate, 0) || approximate <= 0 {
		return nil, fmt.Errorf("multiple_of must be a positive finite number")
	}
	divisor, ok := new(big.Rat).SetString(value.String())
	if !ok {
		return nil, fmt.Errorf("multiple_of is not a decimal number")
	}
	return divisor, nil
}

func matchesMultipleOf(value, bound json.Number) (bool, error) {
	divisor, err := positiveMultipleOf(bound)
	if err != nil {
		return false, err
	}
	number, ok := new(big.Rat).SetString(value.String())
	if !ok {
		return false, fmt.Errorf("value is not numeric")
	}
	return new(big.Rat).Quo(number, divisor).IsInt(), nil
}
