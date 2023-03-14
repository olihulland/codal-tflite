#ifndef TENSORFLOW_LITE_CODAL_H
#define TENSORFLOW_LITE_CODAL_H

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

/**
 * Class for interfacing CODAL with Tensorflow Lite models.
 * 
 * Integrates with CODAL and provides a simple API for running
 * pre-trained Tensorflow Lite models.
 * 
 * @see [Model Creation] (https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train/README.md)
 */
class TfLiteCodal {
    private:
        const tflite::Model * model;
        tflite::MicroInterpreter * interpreter;
        TfLiteTensor * inputTensor;
        TfLiteTensor * outputTensor;

        int kTensorArenaSize;
        uint8_t * tensorArena;

    public:
        TfLiteCodal();

        /**
         * Initialise the TfLiteCodal object with a model.
         * 
         * This method must be called before calling infer().
         * 
         * @see [Model Creation] (https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/train/README.md)
         * 
         * @param model The converted model to use for inference.
         */
        void initialise(const unsigned char* model, int arenaSize);
        
        /**
         * Enum of the data types supported by a tensor for input/output.
         */
        enum TensorType {
            TT_INT32,
            TT_UINT32,
            TT_INT64,
            TT_UINT64,
            TT_FLOAT,
            TT_FLOAT_16,
            TT_FLOAT_64_DOUBLE,
            TT_RAW,
            TT_RAW_CONST,
            TT_UINT8,
            TT_BOOL,
            TT_INT16,
            TT_UINT16,
            TT_COMPLEX64,
            TT_COMPLEX128,
            TT_INT8
        };

        /**
         * Run inference on the model.
         * 
         * @param input pointer to the data to input
         * @param inputType the type of the data being input (also output at the moment)
         * @return void* a pointer to the output data - e.g. can dereference this using *((float*) output) 
         */
        void * infer(void * input, TensorType inputType);

        void * inferArray(void *input, TensorType inputType, int arrayLen);
};

#endif  // TENSORFLOW_LITE_CODAL_H