# Image Downloader

A powerful Python tool for downloading images directly from websites without needing CSV files. This tool can extract image URLs from any website and download them concurrently with robust error handling and progress tracking.

## Features

- **Direct Website Scraping**: Extract image URLs directly from any website
- **Concurrent Downloads**: Download multiple images simultaneously for efficiency
- **Robust Error Handling**: Automatic retries for failed downloads
- **Progress Tracking**: Visual progress bars and real-time statistics
- **Flexible Filtering**: Filter images by size, extension, or custom patterns
- **Resumable Downloads**: Continue interrupted downloads where you left off
- **JavaScript Support**: Option to use Selenium for JavaScript-heavy websites
- **Throttling**: Control download speed to avoid overwhelming servers
- **Comprehensive Logging**: Detailed logs for troubleshooting

## Installation

### Prerequisites

- Python 3.6 or higher
- Required Python packages:
  - requests
  - beautifulsoup4
  - pillow
  - tqdm
  - selenium (optional, for JavaScript-heavy websites)

### Install Dependencies

```bash
pip install requests beautifulsoup4 pillow tqdm
# Optional for JavaScript-heavy websites
pip install selenium webdriver-manager
```

## Usage

### Basic Usage

To download all images from a website:

```bash
python image_downloader.py https://example.com
```

This will download all images to a directory named `downloaded_images` in the current working directory.

### Command-Line Options

#### Output Options

- `-o, --output-dir`: Directory to save downloaded images (default: `downloaded_images`)
- `--create-subdir`: Create a subdirectory based on the website domain

#### Filter Options

- `--min-size`: Minimum image size in bytes (default: 0)
- `--extensions`: Comma-separated list of allowed image extensions (default: `.jpg,.jpeg,.png,.gif,.webp`)
- `--include-regex`: Regular expression pattern that image URLs must match
- `--exclude-regex`: Regular expression pattern that image URLs must not match

#### Download Options

- `-w, --workers`: Number of concurrent download workers (default: 10)
- `--timeout`: Connection timeout in seconds (default: 30)
- `--retries`: Number of retry attempts for failed downloads (default: 3)
- `--throttle`: Throttle downloads to N requests per second (0 for no limit)
- `--no-resume`: Do not resume from previous download state
- `--retry-failed`: Retry previously failed downloads

#### Browser Options

- `--use-selenium`: Use Selenium for JavaScript-heavy websites
- `--wait-time`: Wait time in seconds for JavaScript to load (default: 5)

#### Display Options

- `--no-progress`: Do not show progress bar
- `-q, --quiet`: Quiet mode (minimal output)
- `-v, --verbose`: Verbose mode (detailed output)

### Examples

#### Download images from a specific website

```bash
python image_downloader.py https://example.com
```

#### Download to a custom directory

```bash
python image_downloader.py https://example.com -o my_images
```

#### Download only JPG and PNG images

```bash
python image_downloader.py https://example.com --extensions .jpg,.png
```

#### Download only images larger than 10KB

```bash
python image_downloader.py https://example.com --min-size 10240
```

#### Use 20 concurrent workers for faster downloads

```bash
python image_downloader.py https://example.com -w 20
```

#### Use Selenium for JavaScript-heavy websites

```bash
python image_downloader.py https://example.com --use-selenium
```

#### Throttle downloads to 2 requests per second

```bash
python image_downloader.py https://example.com --throttle 2
```

#### Filter images by URL pattern

```bash
python image_downloader.py https://example.com --include-regex "product|large"
```

#### Retry previously failed downloads

```bash
python image_downloader.py https://example.com --retry-failed
```

## Troubleshooting

### Common Issues

#### No Images Found

If no images are found:
- Check if the website uses JavaScript to load images. Try using the `--use-selenium` option.
- Check if images are loaded from a different domain. The tool only extracts images from the specified website by default.

#### Download Errors

If downloads fail:
- Check your internet connection
- Try increasing the timeout with `--timeout 60`
- Some websites may block automated downloads. Try using the `--throttle` option to slow down requests.

#### Selenium Issues

If using Selenium:
- Ensure you have Chrome installed
- Try increasing the wait time with `--wait-time 10`

### Logs

For detailed logs, use the verbose mode:

```bash
python image_downloader.py https://example.com -v
```

Logs are also saved to `image_downloader.log` in the current directory.

## License

This project is licensed under the MIT License.