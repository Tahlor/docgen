from time import sleep
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import argparse
from pathlib import Path
import requests
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import logging
import pickle

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

class PDFDownloader:
    def __init__(self, save_directory: Path):
        self.save_directory = save_directory

    def download_pdf(self, url: str, save_path: Path) -> None:
        """Download a PDF from a given URL and save to the specified path."""
        response = requests.get(url)
        with save_path.open('wb') as file:
            file.write(response.content)

    def get_pdf_links_from_page(self, url: str, next_button_xpath: str = None) -> list:
        """Retrieve all PDF links from a webpage using undetected_chromedriver."""
        options = uc.ChromeOptions()
        with uc.Chrome(options=options) as driver:
            driver.get(url)
            pdf_links = []
            page_number = 1

            # Check if links file already exists
            links_file = self.save_directory / "links.pkl"
            if links_file.exists():
                with open(links_file, "rb") as f:
                    data = pickle.load(f)
                    pdf_links = data['links']
                    page_number = data['page_number']

                # Navigate to the user-specified page
                for _ in range(page_number - page_number):
                    next_button = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, next_button_xpath)))
                    next_button.click()
                    sleep(5)
                page_number = page_number

            logger.info("Starting to scrape PDF links, waiting 15 seconds for user to change links per page etc...")
            sleep(15)
            while True:
                sleep(8)
                success = False
                while not success:
                    try:
                        pdf_links.extend([link.get_attribute('href') for link in driver.find_elements(By.TAG_NAME, 'a') if
                                          link.get_attribute('href') and link.get_attribute('href').endswith('.pdf')])
                        break
                    except:
                        logger.info("Failed to scrape PDF links, waiting 5 seconds and trying again...")
                        sleep(5)

                # Save links to file
                self.save_links(links_file, pdf_links, page_number)

                if not next_button_xpath:
                    break

                try:
                    next_button = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, next_button_xpath)))
                    next_button.click()
                    page_number += 1
                except:
                    break

        pdf_links = list(set(pdf_links))
        self.save_links(links_file, pdf_links, page_number)
        return pdf_links

    def save_links(self, links_file: Path = None, pdf_links: list = None, page_number: int = None) -> None:
        # Save the links and current page number
        with open(links_file, "wb") as f:
            pickle.dump({'links': pdf_links, 'page_number': page_number}, f)
            logger.info(f"Saved {len(pdf_links)} links to {links_file}")

    def process_organization(self, organization_info: dict) -> None:
        """Main function to download all PDFs from a given webpage."""
        for url in organization_info["urls"]:
            pdf_links = self.get_pdf_links_from_page(url, organization_info.get("xpath"))
            for link in pdf_links:
                file_name = Path(link).name
                self.download_pdf(link, self.save_directory / file_name)
                logger.info(f"Downloaded {file_name}")
        logger.info(f"Downloaded {len(pdf_links)} PDFs to {self.save_directory}")


if __name__ == "__main__":
    o = {
        # "IRS": {"urls": ["https://www.irs.gov/forms-instructions-and-publications"],
        #         "xpath": "//li[@class='pager__item pager__item--next']/a/span[2]"},
        #
        # "OPM": {"urls": ["https://www.opm.gov/forms/optional-forms/",
        #         "https://www.opm.gov/forms/opm-forms",
        #         "https://www.opm.gov/forms/retirement-and-insurance-forms",
        #         "https://www.opm.gov/forms/federal-investigation-forms",
        #         "https://www.opm.gov/forms/federal-employees-group-life-insurance-forms",
        #          "https://www.opm.gov/forms/standard-forms/",],
        #          "xpath": None},
        # "SSA":{"urls":["https://www.ssa.gov/forms/"], "xpath": None},
         "GSA" : {"urls": ["https://www.gsa.gov/forms",
                           "https://www.gsa.gov/forms/obsolete-forms"],
                    "xpath": "//a[@id='DataTables_Table_0_next']"}

         }

    for k, v in o.items():
        save_directory = Path(f"G:/s3/forms/PDF/{k}")
        save_directory.mkdir(parents=True, exist_ok=True)
        downloader = PDFDownloader(save_directory)
        downloader.process_organization(v)
