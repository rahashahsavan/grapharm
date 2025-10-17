#!/usr/bin/env python3
"""
Test runner script for GraphARM.
This script runs all available tests in sequence.
"""

import subprocess
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_test(script_name, description):
    """Run a test script and return success status."""
    logger.info(f"🧪 Running {description}...")
    logger.info("=" * 50)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} PASSED")
            return True
        else:
            logger.error(f"❌ {description} FAILED")
            logger.error(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ {description} TIMED OUT")
        return False
    except Exception as e:
        logger.error(f"❌ {description} FAILED with exception: {e}")
        return False


def main():
    """Run all GraphARM tests."""
    logger.info("🚀 Starting GraphARM Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("quick_test.py", "Quick Component Test"),
        ("test_end_to_end.py", "End-to-End Pipeline Test"),
        ("test_grapharm_complete.py", "Comprehensive Test Suite")
    ]
    
    results = {}
    
    for script, description in tests:
        if os.path.exists(script):
            results[description] = run_test(script, description)
        else:
            logger.warning(f"⚠️  {script} not found, skipping {description}")
            results[description] = False
        
        logger.info("")  # Empty line for readability
    
    # Print summary
    logger.info("=" * 60)
    logger.info("📊 TEST SUITE SUMMARY")
    logger.info("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:30} : {status}")
    
    logger.info("=" * 60)
    logger.info(f"📈 OVERALL RESULT: {passed_tests}/{total_tests} test suites passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! GraphARM is ready for production!")
        logger.info("🚀 You can now run:")
        logger.info("   python train.py --data_dir ./data/ZINC --output_dir ./outputs")
        logger.info("   python generate_molecules.py --checkpoint ./outputs/best_model_model.pt --num_molecules 10000")
        return True
    else:
        logger.warning(f"⚠️  {total_tests - passed_tests} test suites failed.")
        logger.warning("Please check the error messages above and fix the issues.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
