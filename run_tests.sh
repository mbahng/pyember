#!/bin/bash 

run_python_tests() {
    echo "Running Python tests..."
    python test/BaseTensor/integrity_test.py
    python test/BaseTensor/util_test.py
    python test/GradTensor/algebra_test.py

    python test/GradTensor/algebra_test.py
    python test/GradTensor/constructor_test.py
    python test/GradTensor/math_test.py
    python test/GradTensor/util_test.py

    python test/Tensor/algebra_test.py
    python test/Tensor/constructor_test.py
    python test/Tensor/grad_test.py
    python test/Tensor/math_test.py
    python test/Tensor/util_test.py

    return $?
}

# Function to run C++ tests
run_cpp_tests() {
    echo "Running C++ tests..."
    ./aten/build/test/tests
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
