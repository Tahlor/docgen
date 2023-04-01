from pathlib import Path
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)
issued_warnings = set()

class DictionaryLoader:
    """Loads and manages field dictionaries"""
    
    def __init__(self, dictionary_dir: Path):
        """
        Initialize DictionaryLoader
        
        Args:
            dictionary_dir (Path): Directory containing dictionary txt files
        """
        self.dictionary_dir = Path(dictionary_dir)
        self.dictionaries = self._load_all_dictionaries()
        
    def _load_all_dictionaries(self) -> Dict[str, pd.DataFrame]:
        """Load all dictionary files from the directory"""
        dictionaries = {}
        for file_path in self.dictionary_dir.glob("*.txt"):
            field_name = file_path.stem
            try:
                df = pd.read_csv(file_path, sep='|')
                dictionaries[field_name] = df
            except Exception as e:
                logger.warning(f"Failed to load dictionary {file_path}: {e}")
        return dictionaries
    
    def get_random_value(self, field_name: str) -> Optional[Tuple[str, str]]:
        """
        Get a random value-encoding pair for a field
        
        Args:
            field_name (str): Name of the field
            
        Returns:
            Tuple[str, str]: (value, encoding) or None if dictionary not found
        """
        if field_name not in self.dictionaries:
            if field_name not in issued_warnings:
                logger.warning(f"No dictionary found for field {field_name}")
                issued_warnings.add(field_name)
            return None
            
        df = self.dictionaries[field_name]
        row = df.sample(n=1).iloc[0]
        return row['value'], row['encoding'] 