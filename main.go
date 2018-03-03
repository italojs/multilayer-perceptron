package main

import (
	"fmt"
	"github.com/oelmekki/matrix"
	"math"
)

const (
	LEARN_RATE = 0.3
	MOMENTUM   = 1
)

func initializeMatrixs() (inputs matrix.Matrix, weightsInputs matrix.Matrix, weightsHidden matrix.Matrix, outputs matrix.Matrix) {
	inputs, err := matrix.Build(
		matrix.Builder{
			matrix.Row{0, 0},
			matrix.Row{0, 1},
			matrix.Row{1, 0},
			matrix.Row{1, 1},
		},
	)
	if err != nil {
		println(err)
	}

	weightsInputs, err = matrix.Build(
		matrix.Builder{
			matrix.Row{-0.424, -0.740, -0.961},
			matrix.Row{0.358, -0.577, -0.469},
		},
	)
	if err != nil {
		println(err)
	}

	weightsHidden, err = matrix.Build(
		matrix.Builder{
			matrix.Row{-0.017},
			matrix.Row{-0.893},
			matrix.Row{0.148},
		},
	)
	if err != nil {
		println(err)
	}

	outputs, err = matrix.Build(
		matrix.Builder{
			matrix.Row{0},
			matrix.Row{1},
			matrix.Row{1},
			matrix.Row{0},
		},
	)
	if err != nil {
		println(err)
	}

	return
}

func main() {
	var outputLayerSigmoidDerivative matrix.Matrix

	times := 10000000
	inputs, weightsInputs, weightsHidden, outputs := initializeMatrixs()

	for times >= 0 {
		hiddenLayer, err := inputs.DotProduct(weightsInputs)
		if err != nil {
			println(err)
		}
		hiddenLayerSigmoid, err := hiddenLayer.Sigmoid()
		if err != nil {
			println(err)
		}

		outputLayer, err := hiddenLayerSigmoid.DotProduct(weightsHidden)
		if err != nil {
			println(err)
		}
		outputLayerSigmoidDerivative, err = outputLayer.Sigmoid()

		if err != nil {
			println(err)
		}

		outputErrors, err := outputs.Substract(outputLayerSigmoidDerivative)
		// fmt.Println(outputErrors)
		var sum float64
		r := outputErrors.Rows() - 1
		for r >= 0 {
			sum += math.Abs(outputErrors.At(r, 0))
			r--
		}

		//absMean := sum / float64(outputErrors.Rows())
		// fmt.Println("Neural network error: ", absMean)
		outputLayerSigmoidDerivative, err = outputLayer.SigmoidDerivative()
		if err != nil {
			println(err)
		}
		outputLayerDelta, err := outputLayerSigmoidDerivative.MultiplyCells(outputErrors)
		if err != nil {
			println(err)
		}

		trasWeightsHidden, _ := weightsHidden.Transpose()
		outputLayerDelta_WeightsHidden, _ := outputLayerDelta.DotProduct(trasWeightsHidden)
		hiddenLayesSigmoidDerivative, _ := hiddenLayer.SigmoidDerivative()
		hiddenLayerDelta, _ := outputLayerDelta_WeightsHidden.MultiplyCells(hiddenLayesSigmoidDerivative)

		transHiddenLayer, _ := hiddenLayerSigmoid.Transpose()
		newWeightsHidden, _ := transHiddenLayer.DotProduct(outputLayerDelta)
		// fmt.Println(newWeightsHidden)
		// fmt.Println(newWeightsHidden.Rows())
		//need apply momento
		// imagine momento = 1 here
		r = weightsHidden.Rows() - 1
		for r >= 0 {
			value := weightsHidden.At(r, 0)
			newValue := newWeightsHidden.At(r, 0)
			weightsHidden.SetAt(r, 0, value+newValue*LEARN_RATE)
			r--
		}

		transInputLayer, _ := inputs.Transpose()
		newWeightsInputs, _ := transInputLayer.DotProduct(hiddenLayerDelta)

		r = weightsInputs.Rows() - 1
		c := weightsInputs.Cols() - 1
		for r >= 0 {
			for c >= 0{
				value := weightsInputs.At(r, c)
				newValue := newWeightsInputs.At(r, c)
				weightsInputs.SetAt(r, c, value+newValue*LEARN_RATE)
				c--
			}
			r--
		}

		if times == 0{
			fmt.Println("outputLayer.Sigmoid()")
			fmt.Println(outputLayer.Sigmoid())
		}

		times--
	}

}
