#include "tflite_codal.h"

TfLiteCodal::TfLiteCodal() {
    model = nullptr;
    interpreter = nullptr;
    inputTensor = nullptr;
    outputTensor = nullptr;
}

void TfLiteCodal::initialise(const unsigned char * model, int arenaSize) {
    // tflite::InitializeTarget();    // TODO this does nothing - based on system_setup.cc

    kTensorArenaSize = arenaSize;
    tensorArena = new uint8_t[kTensorArenaSize];

    this->model = tflite::GetModel(model);
    if (this->model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf(
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            this->model->version(), TFLITE_SCHEMA_VERSION);
        // TODO handle errors better
        return;
    }

    static tflite::AllOpsResolver resolver;     // TODO could explore using micromutableopresolver

    static tflite::MicroInterpreter static_interpreter(
        this->model, resolver, tensorArena, kTensorArenaSize);
    this->interpreter = &static_interpreter;

    TfLiteStatus allocate_status = this->interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        // TODO handle errors better
        return;
    }

    this->inputTensor = this->interpreter->input(0);
    this->outputTensor = this->interpreter->output(0);
}

void * TfLiteCodal::inferArray(void * input, TensorType inputType, int arrayLen) {
    switch (inputType) {
        case TfLiteCodal::TensorType::TT_FLOAT:
            for (int i = 0; i < arrayLen; i++)
                this->inputTensor->data.f[i] = ((float*) input)[i];
            break;
    }

    // run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed");
        // TODO handle error better
        return nullptr;
    }

    return outputTensor->data.data;
}

void * TfLiteCodal::infer(void * input, TensorType inputType) {
    // set input tensor and find pointer to output tensor
    switch (inputType) {
        case TfLiteCodal::TensorType::TT_INT8:
            this->inputTensor->data.int8[0] = *((int8_t*) input);
            break;
        case TfLiteCodal::TensorType::TT_FLOAT:
            this->inputTensor->data.f[0] = *((float*) input);
            break;
    }

    // run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed");
        // TODO handle error better
        return nullptr;
    }

    return outputTensor->data.data;
}