#!/usr/bin/env python3
"""
Test script for the image downloader.
"""

import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add the current directory to the path so we can import the image_downloader module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_downloader import (
    URLHandler,
    WebScraper,
    ImageExtractor,
    DownloadManager,
    EnhancedDownloadManager,
    ResumableDownloadManager,
    ProgressAwareDownloadManager
)

class TestURLHandler(unittest.TestCase):
    """Test the URLHandler class."""
    
    def setUp(self):
        self.url_handler = URLHandler()
    
    def test_validate_url_valid(self):
        """Test that valid URLs are validated correctly."""
        valid_urls = [
            'http://example.com',
            'https://example.com',
            'http://example.com/path',
            'https://example.com/path?query=value',
            'https://example.com:8080',
            'http://localhost',
            'http://127.0.0.1'
        ]
        
        for url in valid_urls:
            with self.subTest(url=url):
                self.assertTrue(self.url_handler.validate_url(url))
    
    def test_validate_url_invalid(self):
        """Test that invalid URLs are rejected."""
        invalid_urls = [
            'example.com',  # Missing scheme
            'ftp://example.com',  # Wrong scheme
            'http:/example.com',  # Missing slash
            'http:///example.com',  # Too many slashes
            'http://',  # Missing domain
            'http://.',  # Invalid domain
            'http:// example.com',  # Space in URL
            ''  # Empty string
        ]
        
        for url in invalid_urls:
            with self.subTest(url=url):
                self.assertFalse(self.url_handler.validate_url(url))
    
    def test_normalize_url(self):
        """Test URL normalization."""
        test_cases = [
            ('example.com', 'https://example.com'),
            ('http://example.com', 'http://example.com'),
            ('https://example.com', 'https://example.com'),
            ('example.com/path', 'https://example.com/path')
        ]
        
        for input_url, expected_url in test_cases:
            with self.subTest(input_url=input_url):
                self.assertEqual(self.url_handler.normalize_url(input_url), expected_url)
    
    @patch('requests.head')
    def test_is_url_accessible(self, mock_head):
        """Test URL accessibility checking."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response
        
        self.assertTrue(self.url_handler.is_url_accessible('https://example.com'))
        
        # Mock failed response
        mock_response.status_code = 404
        self.assertFalse(self.url_handler.is_url_accessible('https://example.com/not-found'))
        
        # Mock exception
        mock_head.side_effect = Exception('Connection error')
        self.assertFalse(self.url_handler.is_url_accessible('https://nonexistent.example.com'))

class TestImageExtractor(unittest.TestCase):
    """Test the ImageExtractor class."""
    
    def setUp(self):
        self.image_extractor = ImageExtractor()
    
    def test_extract_image_urls_from_html(self):
        """Test extracting image URLs from HTML content."""
        html_content = """
        <html>
        <head>
            <title>Test Page</title>
        </head>
        <body>
            <img src="image1.jpg" alt="Image 1">
            <img src="/image2.png" alt="Image 2">
            <img src="https://example.com/image3.gif" alt="Image 3">
            <img data-src="lazy-image.jpg" alt="Lazy Image">
            <img srcset="small.jpg 320w, medium.jpg 640w, large.jpg 1024w" alt="Responsive Image">
            <div style="background-image: url('background.jpg')"></div>
            <style>
                .header { background-image: url('header.png'); }
            </style>
        </body>
        </html>
        """
        
        base_url = 'https://example.com'
        image_urls = self.image_extractor.extract_image_urls(html_content, base_url)
        
        expected_urls = [
            'https://example.com/image1.jpg',
            'https://example.com/image2.png',
            'https://example.com/image3.gif',
            'https://example.com/lazy-image.jpg',
            'https://example.com/small.jpg',
            'https://example.com/medium.jpg',
            'https://example.com/large.jpg',
            'https://example.com/background.jpg',
            'https://example.com/header.png'
        ]
        
        # Check that all expected URLs are in the result
        for url in expected_urls:
            with self.subTest(url=url):
                self.assertIn(url, image_urls)
        
        # Check that the result has the expected number of URLs
        self.assertEqual(len(image_urls), len(expected_urls))
    
    def test_is_valid_image_url(self):
        """Test validation of image URLs based on extension."""
        valid_urls = [
            'https://example.com/image.jpg',
            'https://example.com/image.jpeg',
            'https://example.com/image.png',
            'https://example.com/image.gif',
            'https://example.com/image.webp',
            'https://example.com/path/to/image.jpg?query=value'
        ]
        
        invalid_urls = [
            'https://example.com/document.pdf',
            'https://example.com/file.txt',
            'https://example.com/image',  # No extension
            'https://example.com/',  # No filename
            None  # None value
        ]
        
        for url in valid_urls:
            with self.subTest(url=url):
                self.assertTrue(self.image_extractor._is_valid_image_url(url))
        
        for url in invalid_urls:
            with self.subTest(url=url):
                self.assertFalse(self.image_extractor._is_valid_image_url(url))

class TestDownloadManager(unittest.TestCase):
    """Test the DownloadManager class."""
    
    def setUp(self):
        # Create a temporary directory for downloads
        self.temp_dir = tempfile.mkdtemp()
        self.download_manager = DownloadManager(output_dir=self.temp_dir, max_workers=2)
    
    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_generate_filename(self):
        """Test filename generation from URLs."""
        test_cases = [
            ('https://example.com/image.jpg', 'image.jpg'),
            ('https://example.com/path/to/image.png', 'image.png'),
            ('https://example.com/image%20with%20spaces.jpg', 'image_with_spaces.jpg'),
            ('https://example.com/image?query=value', None)  # Will generate a hash-based name
        ]
        
        for url, expected_filename in test_cases:
            with self.subTest(url=url):
                filename = self.download_manager._generate_filename(url)
                if expected_filename:
                    self.assertEqual(filename, expected_filename)
                else:
                    self.assertTrue(len(filename) > 0)
                    self.assertTrue('.' in filename)

if __name__ == '__main__':
    unittest.main()