"""
Complete pipeline runner for Urban Carbon Emissions Forecasting
Run this script to execute the entire pipeline:
1. ETL - Generate/load timeseries data
2. Preprocessing & LSTM Training - Train models for all countries + Berlin
3. Ready to run Streamlit app
"""

import os
import sys

def run_etl():
    """Step 1: Run ETL to generate timeseries data"""
    print("\n" + "="*60)
    print("STEP 1: Running ETL Pipeline")
    print("="*60)
    
    if not os.path.exists("etl.py"):
        print("❌ etl.py not found!")
        return False
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "etl.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("❌ ETL failed!")
            print(result.stderr)
            return False
        print("✅ ETL completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error running ETL: {e}")
        return False

def run_training():
    """Step 2: Train LSTM models"""
    print("\n" + "="*60)
    print("STEP 2: Training LSTM Models")
    print("="*60)
    
    if not os.path.exists("preprocessing_and_lstm.py"):
        print("❌ preprocessing_and_lstm.py not found!")
        return False
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "preprocessing_and_lstm.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("❌ Training failed!")
            print(result.stderr)
            return False
        print("✅ Model training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return False

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        "berlin_timeseries.csv",
        "lstm_berlin_model.h5",
        "berlin_scaler.joblib"
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        return False
    return True

def main():
    print("\n" + "="*60)
    print("🌍 Urban Carbon Emissions Forecasting Pipeline")
    print("="*60)
    
    # Check if data already exists
    if os.path.exists("berlin_timeseries.csv") and os.path.exists("europe_timeseries.csv"):
        print("\n✅ Data files already exist. Skipping ETL step.")
        skip_etl = True
    else:
        skip_etl = False
    
    # Run ETL if needed
    if not skip_etl:
        if not run_etl():
            print("\n❌ Pipeline failed at ETL step!")
            return
    
    # Check if models already exist
    if os.path.exists("lstm_berlin_model.h5") and os.path.exists("berlin_scaler.joblib"):
        print("\n✅ Models already exist. Skipping training step.")
        skip_training = True
    else:
        skip_training = False
    
    # Run training if needed
    if not skip_training:
        if not run_training():
            print("\n❌ Pipeline failed at training step!")
            return
    
    # Final check
    if check_requirements():
        print("\n" + "="*60)
        print("✅ Pipeline completed successfully!")
        print("="*60)
        print("\n📊 To run the dashboard, execute:")
        print("   streamlit run app.py")
        print("\n" + "="*60)
    else:
        print("\n❌ Pipeline completed but some files are missing!")
        print("   Please check the errors above.")

if __name__ == "__main__":
    main()
