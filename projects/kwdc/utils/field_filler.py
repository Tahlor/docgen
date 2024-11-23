from typing import Tuple, Optional
import random
from .dictionary_loader import DictionaryLoader

class FieldFiller:
    """Manages field population logic"""
    
    # Fields that should be blank with specific probabilities
    ALWAYS_BLANK = {'SelfExceptionsText'}
    
    BLANK_75_PERCENT = {
        'SelfMilitaryBranchText', 'SelfMilitaryEntryDateText',
        'SelfMilitaryNumberText', 'SelfMilitaryOrganizationText',
        'SelfMilitaryReserveBranchText', 'SelfMilitaryReserveEntryDateText',
        'SelfMilitaryReserveGradeOrganizationText', 'SelfMilitaryReserveGradeText',
        'SelfMilitaryReserveNumberText', 'SelfMilitaryReserveOrganizationText',
        'SelfMilitaryReserveSeparationDateText', 'SelfMilitarySeparationDateText',
        'SelfResidenceMailingAddress'
    }
    
    BLANK_50_PERCENT = {'SelfCharacteristicsText'}
    
    BLANK_25_PERCENT = {
        'SelfOccupation', 'SelfIndustryText', 'SelfEmployerNameText',
        'SelfEmployerPlaceText', 'SelfResidenceStreetAddress'
    }
    
    def __init__(self, dictionary_loader: DictionaryLoader):
        """
        Initialize FieldFiller
        
        Args:
            dictionary_loader (DictionaryLoader): Loader for field dictionaries
        """
        self.dictionary_loader = dictionary_loader
        
    def should_fill(self, field_name: str) -> Tuple[bool, Optional[str]]:
        """
        Determine if a field should be filled and what placeholder to use if blank
        
        Args:
            field_name (str): Name of the field
            
        Returns:
            Tuple[bool, Optional[str]]: (should_fill, placeholder_if_blank)
        """
        if field_name in self.ALWAYS_BLANK:
            return False, None
            
        if field_name in self.BLANK_75_PERCENT and random.random() < 0.75:
            return False, "-" if random.random() < 0.25 else None
            
        if field_name in self.BLANK_50_PERCENT and random.random() < 0.50:
            return False, "-" if random.random() < 0.25 else None
            
        if field_name in self.BLANK_25_PERCENT and random.random() < 0.25:
            return False, "-" if random.random() < 0.25 else None
            
        return True, None
        
    def get_value(self, field_name: str) -> Optional[Tuple[str, str]]:
        """
        Get a value-encoding pair for a field
        
        Args:
            field_name (str): Name of the field
            
        Returns:
            Optional[Tuple[str, str]]: (value, encoding) or None if no dictionary
        """
        return self.dictionary_loader.get_random_value(field_name)