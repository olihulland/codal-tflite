#include "tflite_codal.h"

TfLiteCodal::TfLiteCodal() {
    model = nullptr;
    interpreter = nullptr;
    inputTensor = nullptr;
    outputTensor = nullptr;
    kTensorArenaSize = 2000;        // TODO explore how we can adjust this
    tensorArena = new uint8_t[kTensorArenaSize];
}

void TfLiteCodal::initialise(const unsigned char * model) {
    // tflite::InitializeTarget();    // TODO this does nothing - based on system_setup.cc

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

void * TfLiteCodal::infer(void * input, TensorType inputType) { // TODO explore having a different output type to input type
    // set input tensor and find pointer to output tensor
    void * output = nullptr;
    switch (inputType) {
        case TfLiteCodal::TensorType::TT_INT8:
            this->inputTensor->data.int8[0] = *((int8_t*) input);
            output = &this->outputTensor->data.int8;
            break;
        case TfLiteCodal::TensorType::TT_FLOAT:
            this->inputTensor->data.f[0] = *((float*) input);
            output = &this->outputTensor->data.f[0];
            break;
    }

    // run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed");
        // TODO handle error better
        return nullptr;
    }

    return output;
}