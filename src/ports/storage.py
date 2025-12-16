"""General file storage port interfaces."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from enum import Enum

class IFileStorage(ABC):
    """File storage interface - simple file operations only."""
    
    @abstractmethod
    async def save_file(
        self,
        file_content: bytes,
        filename: str
    ) -> Path:
        """Save file to storage.
        
        Args:
            file_content: File content as bytes
            filename: Filename to save
            
        Returns:
            Path: Path where file was saved
        """
        pass
    
    @abstractmethod
    async def delete_file(self, filename: str) -> bool:
        """Delete file from storage.
        
        Args:
            filename: Filename to delete
            
        Returns:
            bool: True if deleted successfully
        """
        pass
    
    @abstractmethod
    async def file_exists(self, filename: str) -> bool:
        """Check if file exists.
        
        Args:
            filename: Filename to check
            
        Returns:
            bool: True if file exists
        """
        pass
    
    @abstractmethod
    async def list_files(self) -> List[str]:
        """List all files in storage.
        
        Returns:
            List[str]: List of filenames
        """
        pass
    
    @abstractmethod
    async def read_file(self, filename: str) -> bytes:
        """Read file from storage.
        
        Args:
            filename: Filename to read
            
        Returns:
            bytes: File content as bytes
            
        Raises:
            FileNotFoundError: If file does not exist
        """
        pass
