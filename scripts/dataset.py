"""
Dataset loader for HAKS Airbus corrosion prediction challenge.
Loads environment data and corrosion labels for training and testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime


class HAKSDataset:
    """
    Dataset class for loading and processing HAKS Airbus corrosion data.
    
    Attributes:
        data_dir: Path to the directory containing the CSV files
        environment_features: List of feature column names from environment data
    """
    
    def __init__(self, data_dir: str = "haks-airbus-x-ibm-x-aws-2026"):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir: Path to the directory containing the data files
        """
        self.data_dir = Path(data_dir)
        
        # Define environment feature columns (excluding ID and date columns)
        self.environment_features = [
            'total_parking_minutes',
            'metar_temperature_c',
            'metar_relative_humidity',
            'metar_dew_point_c',
            'metar_wind_speed_kn',
            'metar_visibility_mi',
            'metar_hour_precipitation',
            'sea_salt_aerosol_003_05_mixing_ratio',
            'sea_salt_aerosol_05_5_mixing_ratio',
            'sea_salt_aerosol_5_20_mixing_ratio',
            'dust_aerosol_003_055_mixing_ratio',
            'dust_aerosol_055_09_mixing_ratio',
            'dust_aerosol_09_20_mixing_ratio',
            'hydrophilic_organic_matter_aerosol_mixing_ratio',
            'hydrophobic_organic_matter_aerosol_mixing_ratio',
            'hydrophilic_black_carbon_aerosol_mixing_ratio',
            'hydrophobic_black_carbon_aerosol_mixing_ratio',
            'sulphate_aerosol_mixing_ratio',
            'ethane',
            'c3h8',
            'isoprene',
            'carbon_monoxide_mass_mixing_ratio',
            'ozone_mass_mixing_ratio',
            'h2o2',
            'formaldehyde',
            'hno3',
            'nitrogen_monoxide_mass_mixing_ratio',
            'nitrogen_dioxide_mass_mixing_ratio',
            'oh',
            'organic_nitrates',
            'specific_humidity',
            'sulphur_dioxide_mass_mixing_ratio',
            'temperature'
        ]
        
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load training data with environment features and corrosion labels.
        
        Returns:
            X_train: Feature matrix (numpy array) with shape (n_samples, n_features)
            y_train: Target labels (numpy array) with shape (n_samples,) - binary corrosion indicator
            train_df: Full training dataframe with metadata
        """
        # Load environment training data
        env_train_path = self.data_dir / "environment_training.csv"
        env_train = pd.read_csv(env_train_path)
        
        # Load corrosion training data
        corr_train_path = self.data_dir / "corrosions_training.csv"
        corr_train = pd.read_csv(corr_train_path)
        
        # Parse observation date to extract year and month
        corr_train['observation_date'] = pd.to_datetime(corr_train['observation_date'])
        corr_train['year_month'] = corr_train['observation_date'].dt.to_period('M').astype(str)
        
        # Create corrosion indicator (1 = corrosion observed, 0 = no corrosion)
        # For each aircraft-month in environment data, check if corrosion was observed
        env_train['corrosion'] = 0
        
        for idx, row in corr_train.iterrows():
            aircraft_id = row['aircraft_id']
            year_month = row['year_month']
            
            # Mark all months for this aircraft up to observation as having corrosion risk
            mask = (env_train['aircraft_id'] == aircraft_id) & \
                   (env_train['year_month'] <= year_month)
            env_train.loc[mask, 'corrosion'] = 1
        
        # Extract features and labels
        X_train = env_train[self.environment_features].values
        y_train = env_train['corrosion'].values
        
        return X_train, y_train, env_train
    
    def load_test_data(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load test data with environment features.
        
        Returns:
            X_test: Feature matrix (numpy array) with shape (n_samples, n_features)
            test_df: Full test dataframe with metadata
        """
        # Load environment test data
        env_test_path = self.data_dir / "environment_test.csv"
        env_test = pd.read_csv(env_test_path)
        
        # Extract features
        X_test = env_test[self.environment_features].values
        
        return X_test, env_test
    
    def load_sample_submission(self) -> pd.DataFrame:
        """
        Load the sample submission file.
        
        Returns:
            sample_submission: DataFrame with id and corrosion_risk columns
        """
        sample_path = self.data_dir / "sample_submission.csv"
        return pd.read_csv(sample_path)
    
    def get_feature_names(self) -> list:
        """
        Get the list of feature names.
        
        Returns:
            List of feature column names
        """
        return self.environment_features.copy()
    
    def get_dataset_info(self) -> dict:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        X_train, y_train, train_df = self.load_training_data()
        X_test, test_df = self.load_test_data()
        
        info = {
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': len(self.environment_features),
            'n_corrosion_cases': int(y_train.sum()),
            'corrosion_rate': float(y_train.mean()),
            'feature_names': self.environment_features,
            'train_aircraft_count': train_df['aircraft_id'].nunique(),
            'test_aircraft_count': test_df['aircraft_id'].nunique()
        }
        
        return info


def load_haks_data(data_dir: str = "haks-airbus-x-ibm-x-aws-2026") -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load all HAKS data at once.
    
    Args:
        data_dir: Path to the directory containing the data files
    
    Returns:
        X_train: Training features (numpy array)
        y_train: Training labels (numpy array)
        X_test: Test features (numpy array)
        train_df: Training dataframe with metadata
        test_df: Test dataframe with metadata
    """
    dataset = HAKSDataset(data_dir)
    X_train, y_train, train_df = dataset.load_training_data()
    X_test, test_df = dataset.load_test_data()
    
    return X_train, y_train, X_test, train_df, test_df


if __name__ == "__main__":
    # Example usage
    print("Loading HAKS Airbus Corrosion Dataset...")
    
    dataset = HAKSDataset()
    
    # Load training data
    X_train, y_train, train_df = dataset.load_training_data()
    print(f"\nTraining data loaded:")
    print(f"  Features shape: {X_train.shape}")
    print(f"  Labels shape: {y_train.shape}")
    print(f"  Corrosion cases: {y_train.sum()} / {len(y_train)} ({y_train.mean():.2%})")
    
    # Load test data
    X_test, test_df = dataset.load_test_data()
    print(f"\nTest data loaded:")
    print(f"  Features shape: {X_test.shape}")
    
    # Display dataset info
    info = dataset.get_dataset_info()
    print(f"\nDataset Information:")
    print(f"  Number of features: {info['n_features']}")
    print(f"  Training samples: {info['n_train_samples']}")
    print(f"  Test samples: {info['n_test_samples']}")
    print(f"  Training aircraft: {info['train_aircraft_count']}")
    print(f"  Test aircraft: {info['test_aircraft_count']}")
    print(f"  Corrosion rate: {info['corrosion_rate']:.2%}")
    
    print("\nFeature names:")
    for i, feature in enumerate(dataset.get_feature_names()[:10], 1):
        print(f"  {i}. {feature}")
    print(f"  ... and {len(dataset.get_feature_names()) - 10} more features")

# Made with Bob
