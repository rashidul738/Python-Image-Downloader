#!/usr/bin/env python3
"""
Image Downloader - A tool to download images directly from websites.

This script allows users to download images from websites without needing CSV files.
It extracts image URLs directly from the website and downloads them concurrently
with robust error handling and progress tracking.
"""

import os
import sys
import re
import time
import json
import hashlib
import logging
import argparse
import functools
from datetime import datetime
from urllib.parse import urlparse, urljoin, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

# Third-party imports - make sure these are installed
import requests
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm

# Optional Selenium support
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("image_downloader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('image_downloader')

###################
# URL Handler
###################

class URLHandler:
    """Handles URL validation, normalization, and accessibility checking."""
    
    @staticmethod
    def validate_url(url):
        """
        Validate if the given string is a properly formatted URL.
        
        Args:
            url (str): The URL to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Simple regex for URL validation
        url_pattern = re.compile(
            r'^(?:http|https)://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return bool(url_pattern.match(url))
    
    @staticmethod
    def normalize_url(url):
        """
        Normalize the URL by ensuring it has a scheme and handling relative paths.
        
        Args:
            url (str): The URL to normalize
            
        Returns:
            str: The normalized URL
        """
        # Add scheme if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        # Parse the URL to ensure it's well-formed
        parsed = urlparse(url)
        
        # Return the normalized URL
        return parsed.geturl()
    
    @staticmethod
    def is_url_accessible(url):
        """
        Check if the URL is accessible by sending a HEAD request.
        
        Args:
            url (str): The URL to check
            
        Returns:
            bool: True if accessible, False otherwise
        """
        try:
            response = requests.head(url, timeout=10)
            return response.status_code < 400
        except requests.RequestException:
            return False

###################
# Web Scraper
###################

class WebScraper:
    """Retrieves HTML content from websites."""
    
    def __init__(self, use_selenium=False, wait_time=5):
        """
        Initialize the web scraper.
        
        Args:
            use_selenium (bool): Whether to use Selenium for JavaScript-heavy sites
            wait_time (int): Time to wait for JavaScript to load (in seconds)
        """
        self.use_selenium = use_selenium
        self.wait_time = wait_time
        self.driver = None
        
        if use_selenium:
            if not SELENIUM_AVAILABLE:
                raise ImportError("Selenium is not installed. Install it with 'pip install selenium webdriver-manager'")
                
            # Set up headless Chrome
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.chrome.service import Service
                from webdriver_manager.chrome import ChromeDriverManager
                
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--window-size=1920,1080")
                
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
                logger.info("Selenium WebDriver initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Selenium WebDriver: {str(e)}")
                raise
    
    def get_html_content(self, url):
        """
        Retrieve the HTML content from the given URL.
        
        Args:
            url (str): The URL to retrieve content from
            
        Returns:
            str: The HTML content
            
        Raises:
            Exception: If there's an error retrieving the content
        """
        if self.use_selenium and self.driver:
            try:
                logger.info(f"Retrieving content from {url} using Selenium")
                self.driver.get(url)
                # Wait for JavaScript to load
                time.sleep(self.wait_time)
                return self.driver.page_source
            except Exception as e:
                error_msg = f"Error retrieving content with Selenium: {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg)
        else:
            try:
                logger.info(f"Retrieving content from {url} using Requests")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()  # Raise exception for 4XX/5XX responses
                return response.text
            except requests.RequestException as e:
                error_msg = f"Error retrieving content with Requests: {str(e)}"
                logger.error(error_msg)
                raise Exception(error_msg)
    
    def close(self):
        """Close the Selenium driver if it's being used."""
        if self.driver:
            logger.info("Closing Selenium WebDriver")
            self.driver.quit()

###################
# Image Extractor
###################

class ImageExtractor:
    """Extracts image URLs from HTML content."""
    
    def __init__(self, min_size=0, allowed_extensions=None):
        """
        Initialize the image extractor.
        
        Args:
            min_size (int): Minimum image size in bytes (0 for no limit)
            allowed_extensions (list): List of allowed image extensions (None for all)
        """
        self.min_size = min_size
        self.allowed_extensions = allowed_extensions or ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        logger.info(f"Image extractor initialized with min_size={min_size}, allowed_extensions={self.allowed_extensions}")
    
    def extract_image_urls(self, html_content, base_url):
        """
        Extract image URLs from HTML content.
        
        Args:
            html_content (str): The HTML content to parse
            base_url (str): The base URL for resolving relative URLs
            
        Returns:
            list: List of image URLs
        """
        logger.info(f"Extracting image URLs from {base_url}")
        soup = BeautifulSoup(html_content, 'html.parser')
        image_urls = set()  # Use a set to avoid duplicates
        
        # Extract from <img> tags
        for img in soup.find_all('img'):
            try:
                src = img.get('src')
                if src:
                    image_urls.add(self._process_url(src, base_url))
                
                # Also check for data-src attribute (lazy loading)
                data_src = img.get('data-src')
                if data_src:
                    image_urls.add(self._process_url(data_src, base_url))
                    
                # Check srcset attribute for responsive images
                srcset = img.get('srcset')
                if srcset:
                    urls = self._parse_srcset(srcset)
                    for url in urls:
                        image_urls.add(self._process_url(url, base_url))
            except (AttributeError, TypeError):
                # Skip elements that don't support attribute access
                continue
        
        # Extract from CSS background images
        style_tags = soup.find_all('style')
        for style in style_tags:
            try:
                style_content = style.string
                if style_content:
                    urls = self._extract_urls_from_css(style_content)
                    for url in urls:
                        image_urls.add(self._process_url(url, base_url))
            except (AttributeError, TypeError):
                continue
        
        # Extract from inline styles
        elements_with_style = soup.find_all(lambda tag: tag.name is not None and tag.has_attr('style'))
        for element in elements_with_style:
            try:
                style_content = element['style']
                urls = self._extract_urls_from_css(style_content)
                for url in urls:
                    image_urls.add(self._process_url(url, base_url))
            except (AttributeError, TypeError, KeyError):
                continue
        
        # Filter out None values and apply extension filter
        image_urls = [url for url in image_urls if url and self._is_valid_image_url(url)]
        
        logger.info(f"Found {len(image_urls)} image URLs")
        return list(image_urls)
    
    def _process_url(self, url, base_url):
        """
        Process a URL by resolving it against the base URL.
        
        Args:
            url (str): The URL to process
            base_url (str): The base URL
            
        Returns:
            str: The processed URL
        """
        # Skip data URLs
        if url and url.startswith('data:'):
            return None
        
        # Handle relative URLs
        return urljoin(base_url, url)
    
    def _parse_srcset(self, srcset):
        """
        Parse the srcset attribute to extract URLs.
        
        Args:
            srcset (str): The srcset attribute value
            
        Returns:
            list: List of URLs
        """
        urls = []
        for src_item in srcset.split(','):
            # The URL is the first part before any descriptors
            url = src_item.strip().split(' ')[0]
            if url:
                urls.append(url)
        return urls
    
    def _extract_urls_from_css(self, css_text):
        """
        Extract image URLs from CSS text.
        
        Args:
            css_text (str): The CSS text
            
        Returns:
            list: List of URLs
        """
        # Match url() patterns in CSS
        url_pattern = re.compile(r'url\([\'"]?([^\'"()]+)[\'"]?\)')
        return url_pattern.findall(css_text)
    
    def _is_valid_image_url(self, url):
        """
        Check if a URL points to a valid image based on extension.
        
        Args:
            url (str): The URL to check
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not url:
            return False
            
        # Parse the URL to get the path
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        # Check if the path ends with an allowed extension
        return any(path.endswith(ext) for ext in self.allowed_extensions)

###################
# Retry Decorator
###################

def retry(max_attempts=3, delay=1, backoff=2, exceptions=(Exception,)):
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts (int): Maximum number of retry attempts
        delay (float): Initial delay between retries in seconds
        backoff (float): Backoff multiplier (e.g. 2 means delay doubles each retry)
        exceptions (tuple): Exceptions to catch and retry
        
    Returns:
        function: Decorated function with retry logic
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = max_attempts, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(f"Retry: {func.__name__} failed with {str(e)}. Retrying in {mdelay} seconds...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)  # Last attempt
        return wrapper
    return decorator

###################
# Progress Tracker
###################

class ProgressTracker:
    """Tracks download progress with a progress bar."""
    
    def __init__(self, total=0, desc="Downloading", unit="img"):
        """
        Initialize the progress tracker.
        
        Args:
            total (int): Total number of items
            desc (str): Description for the progress bar
            unit (str): Unit name for the progress bar
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.progress_bar = None
        self.start_time = None
        self.stats = {
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'total_size': 0
        }
    
    def start(self, total=None):
        """
        Start the progress tracking.
        
        Args:
            total (int, optional): Override the total number of items
        """
        if total is not None:
            self.total = total
            
        self.start_time = time.time()
        self.progress_bar = tqdm(
            total=self.total,
            desc=self.desc,
            unit=self.unit,
            unit_scale=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
    
    def update(self, result):
        """
        Update the progress based on a download result.
        
        Args:
            result (dict): Download result information
        """
        if not self.progress_bar:
            return
            
        self.progress_bar.update(1)
        
        # Update statistics
        if result.get('success', False):
            if result.get('status') == 'skipped':
                self.stats['skipped'] += 1
            else:
                self.stats['successful'] += 1
                self.stats['total_size'] += result.get('size', 0)
        else:
            self.stats['failed'] += 1
            
        # Update progress bar description with stats
        self.progress_bar.set_postfix(
            success=self.stats['successful'],
            failed=self.stats['failed'],
            skipped=self.stats['skipped']
        )
    
    def finish(self):
        """Finish the progress tracking and display summary."""
        if not self.progress_bar:
            return
            
        self.progress_bar.close()
        
        # Calculate elapsed time
        elapsed = time.time() - (self.start_time or time.time())
        
        # Print summary
        print("\nDownload Summary:")
        print(f"Total images: {self.total}")
        print(f"Successfully downloaded: {self.stats['successful']}")
        print(f"Skipped (already exists): {self.stats['skipped']}")
        print(f"Failed: {self.stats['failed']}")
        
        # Print size information
        if self.stats['total_size'] > 0:
            size_mb = self.stats['total_size'] / (1024 * 1024)
            print(f"Total downloaded size: {size_mb:.2f} MB")
            
        # Print time information
        print(f"Total time: {elapsed:.2f} seconds")
        if self.stats['successful'] > 0:
            avg_time = elapsed / self.stats['successful']
            print(f"Average time per image: {avg_time:.2f} seconds")
            
        # Print timestamp
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

###################
# Download Manager
###################

class DownloadManager:
    """Manages concurrent downloads with error handling."""
    
    def __init__(self, output_dir="downloaded_images", max_workers=10, timeout=30):
        """
        Initialize the download manager.
        
        Args:
            output_dir (str): Directory to save downloaded images
            max_workers (int): Maximum number of concurrent downloads
            timeout (int): Timeout for download requests in seconds
        """
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.timeout = timeout
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Statistics
        self.total_images = 0
        self.successful_downloads = 0
        self.failed_downloads = 0
        self.retried_downloads = 0
        
        # Session for connection pooling
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        logger.info(f"Download manager initialized with output_dir={output_dir}, max_workers={max_workers}")
    
    def download_images(self, image_urls, progress_callback=None):
        """
        Download images concurrently.
        
        Args:
            image_urls (list): List of image URLs to download
            progress_callback (function, optional): Callback for progress updates
            
        Returns:
            dict: Statistics about the download process
        """
        self.total_images = len(image_urls)
        logger.info(f"Starting download of {self.total_images} images")
        
        # Create a thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit download tasks
            future_to_url = {
                executor.submit(self._download_image, url): url 
                for url in image_urls
            }
            
            # Process completed downloads
            for i, future in enumerate(as_completed(future_to_url)):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result['success']:
                        self.successful_downloads += 1
                    else:
                        self.failed_downloads += 1
                        
                    # Call progress callback if provided
                    if progress_callback:
                        progress = (i + 1) / self.total_images
                        progress_callback(progress, result)
                        
                except Exception as e:
                    self.failed_downloads += 1
                    logger.error(f"Error processing download result for {url}: {str(e)}")
                    if progress_callback:
                        progress = (i + 1) / self.total_images
                        progress_callback(progress, {
                            'success': False,
                            'url': url,
                            'error': str(e)
                        })
        
        # Return statistics
        return {
            'total': self.total_images,
            'successful': self.successful_downloads,
            'failed': self.failed_downloads,
            'retried': self.retried_downloads
        }
    
    def _download_image(self, url):
        """
        Download a single image.
        
        Args:
            url (str): URL of the image to download
            
        Returns:
            dict: Result of the download operation
        """
        try:
            # Generate a filename for the image
            filename = self._generate_filename(url)
            filepath = os.path.join(self.output_dir, filename)
            
            # Check if file already exists
            if os.path.exists(filepath):
                logger.info(f"Skipping {url} - File already exists")
                return {
                    'success': True,
                    'url': url,
                    'filepath': filepath,
                    'status': 'skipped',
                    'message': 'File already exists'
                }
            
            # Download the image
            logger.info(f"Downloading {url}")
            response = self.session.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            # Check if it's actually an image
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                logger.warning(f"Not an image: {url} (Content-Type: {content_type})")
                return {
                    'success': False,
                    'url': url,
                    'error': f'Not an image (Content-Type: {content_type})'
                }
            
            # Save the image
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify the image
            try:
                img = Image.open(filepath)
                img.verify()  # Verify it's a valid image
                
                logger.info(f"Successfully downloaded {url}")
                return {
                    'success': True,
                    'url': url,
                    'filepath': filepath,
                    'status': 'downloaded',
                    'size': os.path.getsize(filepath),
                    'dimensions': img.size
                }
            except Exception as e:
                # Not a valid image, remove the file
                logger.warning(f"Invalid image: {url} - {str(e)}")
                os.remove(filepath)
                return {
                    'success': False,
                    'url': url,
                    'error': f'Invalid image: {str(e)}'
                }
                
        except requests.RequestException as e:
            logger.error(f"Download error: {url} - {str(e)}")
            return {
                'success': False,
                'url': url,
                'error': f'Download error: {str(e)}'
            }
        except Exception as e:
            logger.error(f"Unexpected error: {url} - {str(e)}")
            return {
                'success': False,
                'url': url,
                'error': f'Unexpected error: {str(e)}'
            }
    
    def _generate_filename(self, url):
        """
        Generate a filename for an image URL.
        
        Args:
            url (str): The image URL
            
        Returns:
            str: The generated filename
        """
        # Try to extract filename from URL
        parsed_url = urlparse(url)
        path = unquote(parsed_url.path)
        filename = os.path.basename(path)
        
        # If filename is empty or doesn't have an extension, generate one
        if not filename or '.' not in filename:
            # Generate a hash of the URL
            url_hash = hashlib.md5(url.encode()).hexdigest()
            
            # Try to determine extension from Content-Type
            try:
                response = self.session.head(url, timeout=5)
                content_type = response.headers.get('Content-Type', '')
                
                if content_type == 'image/jpeg':
                    ext = '.jpg'
                elif content_type == 'image/png':
                    ext = '.png'
                elif content_type == 'image/gif':
                    ext = '.gif'
                elif content_type == 'image/webp':
                    ext = '.webp'
                elif content_type == 'image/bmp':
                    ext = '.bmp'
                else:
                    ext = '.jpg'  # Default to jpg
                    
                filename = f"{url_hash}{ext}"
            except:
                # If we can't determine the type, use a default
                filename = f"{url_hash}.jpg"
        
        # Ensure filename is valid and not too long
        filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        if len(filename) > 100:
            # Truncate long filenames
            name, ext = os.path.splitext(filename)
            filename = f"{name[:90]}{ext}"
            
        return filename

###################
# Enhanced Download Manager
###################

class EnhancedDownloadManager(DownloadManager):
    """Enhanced download manager with retry capabilities."""
    
    def __init__(self, output_dir="downloaded_images", max_workers=10, timeout=30, 
                 max_retries=3, retry_delay=1):
        """
        Initialize the enhanced download manager with retry capabilities.
        
        Args:
            output_dir (str): Directory to save downloaded images
            max_workers (int): Maximum number of concurrent downloads
            timeout (int): Timeout for download requests in seconds
            max_retries (int): Maximum number of retry attempts
            retry_delay (float): Initial delay between retries in seconds
        """
        super().__init__(output_dir, max_workers, timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(f"Enhanced download manager initialized with max_retries={max_retries}, retry_delay={retry_delay}")
    
    @retry(max_attempts=3, delay=1, backoff=2, 
           exceptions=(requests.RequestException, ConnectionError, TimeoutError))
    def _fetch_image(self, url):
        """
        Fetch an image with retry logic.
        
        Args:
            url (str): URL of the image to fetch
            
        Returns:
            requests.Response: The response object
            
        Raises:
            requests.RequestException: If the request fails
        """
        response = self.session.get(url, timeout=self.timeout, stream=True)
        response.raise_for_status()
        return response
    
    def _download_image(self, url):
        """
        Download a single image with enhanced error handling.
        
        Args:
            url (str): URL of the image to download
            
        Returns:
            dict: Result of the download operation
        """
        try:
            # Generate a filename for the image
            filename = self._generate_filename(url)
            filepath = os.path.join(self.output_dir, filename)
            
            # Check if file already exists
            if os.path.exists(filepath):
                logger.info(f"Skipping {url} - File already exists")
                return {
                    'success': True,
                    'url': url,
                    'filepath': filepath,
                    'status': 'skipped',
                    'message': 'File already exists'
                }
            
            # Download the image with retry
            try:
                logger.info(f"Downloading {url}")
                response = self._fetch_image(url)
                self.retried_downloads += 1
                
                # Check if it's actually an image
                content_type = response.headers.get('Content-Type', '')
                if not content_type.startswith('image/'):
                    logger.warning(f"Not an image: {url} (Content-Type: {content_type})")
                    return {
                        'success': False,
                        'url': url,
                        'error': f'Not an image (Content-Type: {content_type})'
                    }
                
                # Save the image
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Verify the image
                try:
                    img = Image.open(filepath)
                    img.verify()  # Verify it's a valid image
                    
                    logger.info(f"Successfully downloaded {url}")
                    return {
                        'success': True,
                        'url': url,
                        'filepath': filepath,
                        'status': 'downloaded',
                        'size': os.path.getsize(filepath),
                        'dimensions': img.size
                    }
                except Exception as e:
                    # Not a valid image, remove the file
                    logger.warning(f"Invalid image: {url} - {str(e)}")
                    os.remove(filepath)
                    return {
                        'success': False,
                        'url': url,
                        'error': f'Invalid image: {str(e)}'
                    }
                    
            except requests.RequestException as e:
                logger.error(f"Download failed: {url} - {str(e)}")
                return {
                    'success': False,
                    'url': url,
                    'error': f'Download error: {str(e)}'
                }
                
        except Exception as e:
            logger.error(f"Unexpected error: {url} - {str(e)}")
            return {
                'success': False,
                'url': url,
                'error': f'Unexpected error: {str(e)}'
            }

###################
# Throttled Download Manager
###################

class ThrottledDownloadManager(EnhancedDownloadManager):
    """Download manager with throttling to limit request rate."""
    
    def __init__(self, output_dir="downloaded_images", max_workers=10, timeout=30, 
                 max_retries=3, retry_delay=1, requests_per_second=5):
        """
        Initialize the throttled download manager.
        
        Args:
            output_dir (str): Directory to save downloaded images
            max_workers (int): Maximum number of concurrent downloads
            timeout (int): Timeout for download requests in seconds
            max_retries (int): Maximum number of retry attempts
            retry_delay (float): Initial delay between retries in seconds
            requests_per_second (float): Maximum number of requests per second
        """
        super().__init__(output_dir, max_workers, timeout, max_retries, retry_delay)
        self.delay = 1.0 / requests_per_second
        self.last_request_time = 0
        
        logger.info(f"Throttled download manager initialized with requests_per_second={requests_per_second}")
    
    def _download_image(self, url):
        """
        Download a single image with throttling.
        
        Args:
            url (str): URL of the image to download
            
        Returns:
            dict: Result of the download operation
        """
        # Implement throttling
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.delay:
            time.sleep(self.delay - time_since_last_request)
            
        self.last_request_time = time.time()
        
        # Call the parent method to perform the actual download
        return super()._download_image(url)

###################
# Resumable Download Manager
###################

class ResumableDownloadManager(EnhancedDownloadManager):
    """Download manager with resume capability."""
    
    def __init__(self, output_dir="downloaded_images", max_workers=10, timeout=30, 
                 max_retries=3, retry_delay=1, state_file="download_state.json"):
        """
        Initialize the resumable download manager.
        
        Args:
            output_dir (str): Directory to save downloaded images
            max_workers (int): Maximum number of concurrent downloads
            timeout (int): Timeout for download requests in seconds
            max_retries (int): Maximum number of retry attempts
            retry_delay (float): Initial delay between retries in seconds
            state_file (str): File to save download state
        """
        super().__init__(output_dir, max_workers, timeout, max_retries, retry_delay)
        self.state_file = state_file
        self.downloaded_urls = set()
        self.failed_urls = set()
        
        # Load previous state if exists
        self._load_state()
        
        logger.info(f"Resumable download manager initialized with state_file={state_file}")
    
    def _load_state(self):
        """Load the previous download state if it exists."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.downloaded_urls = set(state.get('downloaded', []))
                    self.failed_urls = set(state.get('failed', []))
                    logger.info(f"Loaded state: {len(self.downloaded_urls)} downloaded, {len(self.failed_urls)} failed")
        except Exception as e:
            logger.warning(f"Failed to load state: {str(e)}")
    
    def _save_state(self):
        """Save the current download state."""
        try:
            state = {
                'downloaded': list(self.downloaded_urls),
                'failed': list(self.failed_urls)
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
            logger.info(f"Saved state: {len(self.downloaded_urls)} downloaded, {len(self.failed_urls)} failed")
        except Exception as e:
            logger.warning(f"Failed to save state: {str(e)}")
    
    def download_images(self, image_urls, progress_callback=None, resume=True):
        """
        Download images concurrently with resume capability.
        
        Args:
            image_urls (list): List of image URLs to download
            progress_callback (function, optional): Callback for progress updates
            resume (bool): Whether to resume from previous state
            
        Returns:
            dict: Statistics about the download process
        """
        # Filter out already downloaded URLs if resuming
        if resume:
            image_urls = [url for url in image_urls if url not in self.downloaded_urls]
            logger.info(f"Resuming download: {len(image_urls)} images remaining")
        
        self.total_images = len(image_urls)
        
        # Create a thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit download tasks
            future_to_url = {
                executor.submit(self._download_image, url): url
                for url in image_urls
            }
            
            # Process completed downloads
            for i, future in enumerate(as_completed(future_to_url)):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result['success']:
                        self.successful_downloads += 1
                        self.downloaded_urls.add(url)
                    else:
                        self.failed_downloads += 1
                        self.failed_urls.add(url)
                        
                    # Call progress callback if provided
                    if progress_callback:
                        progress = (i + 1) / self.total_images
                        progress_callback(progress, result)
                        
                    # Save state periodically (every 10 downloads)
                    if (i + 1) % 10 == 0:
                        self._save_state()
                        
                except Exception as e:
                    self.failed_downloads += 1
                    self.failed_urls.add(url)
                    if progress_callback:
                        progress = (i + 1) / self.total_images
                        progress_callback(progress, {
                            'success': False,
                            'url': url,
                            'error': str(e)
                        })
        
        # Save final state
        self._save_state()
        
        # Return statistics
        return {
            'total': self.total_images,
            'successful': self.successful_downloads,
            'failed': self.failed_downloads,
            'retried': self.retried_downloads
        }
    
    def retry_failed(self, progress_callback=None):
        """
        Retry previously failed downloads.
        
        Args:
            progress_callback (function, optional): Callback for progress updates
            
        Returns:
            dict: Statistics about the retry process
        """
        failed_urls = list(self.failed_urls)
        logger.info(f"Retrying {len(failed_urls)} failed downloads")
        
        # Clear failed URLs before retrying
        self.failed_urls = set()
        
        # Reset statistics for this operation
        self.total_images = len(failed_urls)
        self.successful_downloads = 0
        self.failed_downloads = 0
        self.retried_downloads = 0
        
        # Download the failed URLs
        result = self.download_images(failed_urls, progress_callback, resume=False)
        
        return result

###################
# Progress-Aware Download Manager
###################

class ProgressAwareDownloadManager(ResumableDownloadManager):
    """Download manager with integrated progress tracking."""
    
    def __init__(self, output_dir="downloaded_images", max_workers=10, timeout=30,
                 max_retries=3, retry_delay=1, state_file="download_state.json"):
        """
        Initialize the progress-aware download manager.
        
        Args:
            output_dir (str): Directory to save downloaded images
            max_workers (int): Maximum number of concurrent downloads
            timeout (int): Timeout for download requests in seconds
            max_retries (int): Maximum number of retry attempts
            retry_delay (float): Initial delay between retries in seconds
            state_file (str): File to save download state
        """
        super().__init__(output_dir, max_workers, timeout, max_retries, retry_delay, state_file)
        
        # Set up progress tracking
        self.progress_tracker = None
        
        logger.info("Progress-aware download manager initialized")
    
    def download_images(self, image_urls, progress_callback=None, resume=True, show_progress=True):
        """
        Download images concurrently with progress tracking.
        
        Args:
            image_urls (list): List of image URLs to download
            progress_callback (function, optional): Callback for progress updates
            resume (bool): Whether to resume from previous state
            show_progress (bool): Whether to show progress bar
            
        Returns:
            dict: Statistics about the download process
        """
        # Initialize progress tracking
        if show_progress:
            self.progress_tracker = ProgressTracker()
        
        # Create a combined progress callback
        original_callback = progress_callback
        
        def combined_callback(progress, result):
            if self.progress_tracker:
                self.progress_tracker.update(result)
            if original_callback:
                original_callback(progress, result)
        
        # Filter out already downloaded URLs if resuming
        if resume:
            image_urls = [url for url in image_urls if url not in self.downloaded_urls]
            logger.info(f"Resuming download: {len(image_urls)} images remaining")
        
        self.total_images = len(image_urls)
        
        # Start progress tracking
        if self.progress_tracker:
            self.progress_tracker.start(self.total_images)
        
        # Call parent method to perform the download
        result = super().download_images(image_urls, combined_callback, resume)
        
        # Finish progress tracking
        if self.progress_tracker:
            self.progress_tracker.finish()
        
        return result

###################
# Command-Line Interface
###################

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Download images from websites directly.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        'url',
        help='URL of the website to download images from'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '-o', '--output-dir',
        default='downloaded_images',
        help='Directory to save downloaded images'
    )
    output_group.add_argument(
        '--create-subdir',
        action='store_true',
        help='Create a subdirectory based on the website domain'
    )
    
    # Filter options
    filter_group = parser.add_argument_group('Filter Options')
    filter_group.add_argument(
        '--min-size',
        type=int,
        default=0,
        help='Minimum image size in bytes'
    )
    filter_group.add_argument(
        '--extensions',
        default='.jpg,.jpeg,.png,.gif,.webp',
        help='Comma-separated list of allowed image extensions'
    )
    filter_group.add_argument(
        '--include-regex',
        help='Regular expression pattern that image URLs must match'
    )
    filter_group.add_argument(
        '--exclude-regex',
        help='Regular expression pattern that image URLs must not match'
    )
    
    # Download options
    download_group = parser.add_argument_group('Download Options')
    download_group.add_argument(
        '-w', '--workers',
        type=int,
        default=10,
        help='Number of concurrent download workers'
    )
    download_group.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Connection timeout in seconds'
    )
    download_group.add_argument(
        '--retries',
        type=int,
        default=3,
        help='Number of retry attempts for failed downloads'
    )
    download_group.add_argument(
        '--throttle',
        type=float,
        default=0,
        help='Throttle downloads to N requests per second (0 for no limit)'
    )
    download_group.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume from previous download state'
    )
    download_group.add_argument(
        '--retry-failed',
        action='store_true',
        help='Retry previously failed downloads'
    )
    
    # Browser options
    browser_group = parser.add_argument_group('Browser Options')
    browser_group.add_argument(
        '--use-selenium',
        action='store_true',
        help='Use Selenium for JavaScript-heavy websites'
    )
    browser_group.add_argument(
        '--wait-time',
        type=int,
        default=5,
        help='Wait time in seconds for JavaScript to load (with Selenium)'
    )
    
    # Display options
    display_group = parser.add_argument_group('Display Options')
    display_group.add_argument(
        '--no-progress',
        action='store_true',
        help='Do not show progress bar'
    )
    display_group.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet mode (minimal output)'
    )
    display_group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose mode (detailed output)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process extensions
    args.extensions = [ext.strip() for ext in args.extensions.split(',')]
    
    # Validate arguments
    if args.throttle < 0:
        parser.error("--throttle must be greater than or equal to 0")
    
    if args.min_size < 0:
        parser.error("--min-size must be greater than or equal to 0")
    
    if args.workers < 1:
        parser.error("--workers must be greater than 0")
    
    return args

###################
# Main Application
###################

class ImageDownloaderApp:
    """Main application class for the image downloader."""
    
    def __init__(self, args):
        """
        Initialize the image downloader application.
        
        Args:
            args (argparse.Namespace): Command-line arguments
        """
        self.args = args
        
        # Set up logging
        log_level = logging.DEBUG if args.verbose else logging.INFO
        if args.quiet:
            log_level = logging.WARNING
            
        # Update root logger level
        logger.setLevel(log_level)
        for handler in logger.handlers:
            handler.setLevel(log_level)
            
        logger.info("Initializing Image Downloader")
        
        # Determine output directory
        self.output_dir = args.output_dir
        if args.create_subdir:
            domain = urlparse(args.url).netloc
            self.output_dir = os.path.join(args.output_dir, domain)
            
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self.url_handler = URLHandler()
        
        # Initialize download manager
        # Always use ProgressAwareDownloadManager as the base
        self.download_manager = ProgressAwareDownloadManager(
            output_dir=self.output_dir,
            max_workers=args.workers,
            timeout=args.timeout,
            max_retries=args.retries
        )
        
        # Apply throttling if needed
        if args.throttle > 0:
            # Create a throttled version that wraps the original download manager's methods
            original_download_image = self.download_manager._download_image
            
            def throttled_download_image(url):
                # Implement throttling
                current_time = time.time()
                time_since_last_request = current_time - getattr(self.download_manager, 'last_request_time', 0)
                delay = 1.0 / args.throttle
                
                if time_since_last_request < delay:
                    time.sleep(delay - time_since_last_request)
                    
                self.download_manager.last_request_time = time.time()
                
                # Call the original method
                return original_download_image(url)
            
            # Replace the method with the throttled version
            self.download_manager._download_image = throttled_download_image
            logger.info(f"Applied throttling: {args.throttle} requests per second")
            
        # Initialize image extractor
        self.image_extractor = ImageExtractor(
            min_size=args.min_size,
            allowed_extensions=args.extensions
        )
    
    def run(self):
        """
        Run the image downloader application.
        
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        try:
            # Validate URL
            if not self.url_handler.validate_url(self.args.url):
                logger.error(f"Invalid URL: {self.args.url}")
                return 1
                
            # Normalize URL
            url = self.url_handler.normalize_url(self.args.url)
            
            # Check if URL is accessible
            if not self.url_handler.is_url_accessible(url):
                logger.error(f"URL is not accessible: {url}")
                return 1
                
            # Initialize web scraper
            web_scraper = WebScraper(
                use_selenium=self.args.use_selenium,
                wait_time=self.args.wait_time
            )
            
            try:
                # Get HTML content
                logger.info(f"Retrieving content from {url}")
                html_content = web_scraper.get_html_content(url)
                
                # Extract image URLs
                logger.info("Extracting image URLs")
                image_urls = self.image_extractor.extract_image_urls(html_content, url)
                
                # Apply regex filters if specified
                if self.args.include_regex:
                    pattern = re.compile(self.args.include_regex)
                    image_urls = [url for url in image_urls if pattern.search(url)]
                    
                if self.args.exclude_regex:
                    pattern = re.compile(self.args.exclude_regex)
                    image_urls = [url for url in image_urls if not pattern.search(url)]
                
                # Log number of images found
                logger.info(f"Found {len(image_urls)} images")
                
                if not image_urls:
                    logger.warning("No images found to download")
                    return 0
                
                # Download images
                logger.info(f"Downloading {len(image_urls)} images to {self.output_dir}")
                
                # Handle retry-failed option
                if self.args.retry_failed and hasattr(self.download_manager, 'retry_failed'):
                    logger.info("Retrying previously failed downloads")
                    stats = self.download_manager.retry_failed()
                else:
                    # Normal download
                    stats = self.download_manager.download_images(
                        image_urls,
                        progress_callback=None,
                        resume=not self.args.no_resume,
                        show_progress=not self.args.no_progress
                    )
                
                # Log statistics
                logger.info(f"Download complete: {stats['successful']} successful, {stats['failed']} failed")
                
                return 0
                
            finally:
                # Clean up resources
                if web_scraper:
                    web_scraper.close()
                    
        except KeyboardInterrupt:
            logger.info("Download interrupted by user")
            return 130
            
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            if self.args.verbose:
                import traceback
                traceback.print_exc()
            return 1

def main():
    """Main entry point for the application."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create and run the application
    app = ImageDownloaderApp(args)
    return app.run()

if __name__ == "__main__":
    sys.exit(main())