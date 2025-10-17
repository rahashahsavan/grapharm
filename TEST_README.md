# GraphARM Test Suite

This directory contains comprehensive test scripts for the GraphARM model implementation.

## Test Scripts

### 1. `quick_test.py` - Quick Component Test
**Purpose**: Fast test of basic components without data loading
**Duration**: ~30 seconds
**What it tests**:
- Model initialization
- Forward pass through networks
- Basic masking operations
- Simple molecule generation

**Run**: `python quick_test.py`

### 2. `test_end_to_end.py` - End-to-End Pipeline Test
**Purpose**: Complete pipeline test from training to generation
**Duration**: ~2-3 minutes
**What it tests**:
- ZINC250k data loading
- Model training (3 epochs)
- Molecule generation
- Model saving/loading
- Batch generation

**Run**: `python test_end_to_end.py`

### 3. `test_grapharm_complete.py` - Comprehensive Test Suite
**Purpose**: Thorough testing of all components
**Duration**: ~5-10 minutes
**What it tests**:
- Data loading and preprocessing
- NodeMasking functionality
- Model initialization and forward pass
- Training procedure
- Diffusion trajectory generation
- Molecule generation
- Batch generation
- Model saving/loading
- Performance metrics

**Run**: `python test_grapharm_complete.py`

### 4. `run_tests.py` - Test Runner
**Purpose**: Runs all tests in sequence
**Duration**: ~10-15 minutes
**What it does**:
- Executes all test scripts
- Provides summary of results
- Gives recommendations for next steps

**Run**: `python run_tests.py`

## Quick Start

### Option 1: Quick Test (Recommended for first run)
```bash
python quick_test.py
```

### Option 2: Complete Test Suite
```bash
python run_tests.py
```

### Option 3: Individual Tests
```bash
# Quick component test
python quick_test.py

# End-to-end pipeline test
python test_end_to_end.py

# Comprehensive test suite
python test_grapharm_complete.py
```

## Expected Outputs

### Successful Test Output
```
✅ GraphARM is working correctly!
🚀 Ready for full training and generation!
```

### Test Results
- **Quick Test**: Tests basic functionality
- **End-to-End Test**: Tests complete pipeline
- **Comprehensive Test**: Tests all components with detailed logging

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - The tests use small models and batches
   - If still failing, reduce batch sizes in test scripts

2. **Data Loading Issues**
   - ZINC250k dataset will be downloaded automatically
   - Ensure internet connection for first run

3. **Import Errors**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

### Debug Mode

For detailed debugging, modify the logging level in test scripts:
```python
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
```

## Test Coverage

The test suite covers:

- ✅ **Data Loading**: ZINC250k dataset
- ✅ **Model Architecture**: Diffusion ordering and denoising networks
- ✅ **Training**: REINFORCE algorithm and loss computation
- ✅ **Generation**: Autoregressive molecule generation
- ✅ **Utilities**: NodeMasking and graph operations
- ✅ **Performance**: Speed and memory usage
- ✅ **Integration**: End-to-end pipeline

## Next Steps

After successful tests:

1. **Train the model**:
   ```bash
   python train.py --data_dir ./data/ZINC --output_dir ./outputs
   ```

2. **Generate molecules**:
   ```bash
   python generate_molecules.py \
       --checkpoint ./outputs/best_model_model.pt \
       --num_molecules 10000 \
       --output_dir ./generated_molecules
   ```

## Notes

- Tests use small models and datasets for speed
- Real training will use larger models and full dataset
- Test outputs are saved in `./test_outputs/` directory
- All tests are designed to run on both CPU and GPU
