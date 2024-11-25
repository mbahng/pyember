#!/bin/bash 

run_python_tests() {
    echo "Running Python tests..."
    python test/tensor/io.py
    python test/tensor/gradtensor.py
    python test/tensor/tensor.py
    python test/tensor/backprop.py
    return $?
}

# Function to run C++ tests
run_cpp_tests() {
    echo "Running C++ tests..."
    ./build/aten/tests
    return $?
}

# Check if an argument was provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide an argument (python/cpp/all)"
    echo "Usage: ./run_tests.sh [python|cpp|all]"
    exit 1
fi

# Process based on argument
case "$1" in
    "python")
        run_python_tests
        exit $?
        ;;
    "cpp")
        run_cpp_tests
        exit $?
        ;;
    "all")
        echo "Running all tests..."
        # Run both sets of tests, exit with failure if either fails
        run_python_tests
        PYTHON_RESULT=$?
        run_cpp_tests
        CPP_RESULT=$?
        
        # If either test failed, exit with failure
        if [ $PYTHON_RESULT -ne 0 ] || [ $CPP_RESULT -ne 0 ]; then
            exit 1
        fi
        exit 0
        ;;
    *)
        echo "Error: Invalid argument '$1'"
        echo "Usage: ./run_tests.sh [python|cpp|all]"
        exit 1
        ;;
esac
