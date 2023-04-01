from time import sleep
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
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

    def load_links_from_pickle(self, links_file: Path) -> tuple:
        """Load links and page number from pickle file."""
        with open(links_file, "rb") as f:
            data = pickle.load(f)
            return data['links'], data['page_number']
        return [], 1

    def navigate_to_page(self, driver, next_button_xpath: str, page_number: int) -> None:
        """Navigate to a specific page number."""
        for _ in range(page_number - 1):
            next_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, next_button_xpath)))
            next_button.click()
            sleep(5)

    def get_gsa_links_and_metadata(self, driver, table_xpath: str) -> list:
        """Retrieve GSA specific links and metadata."""
        rows = driver.find_elements(By.XPATH, table_xpath)
        links_metadata = []
        for row in rows:
            metadata = ' '.join([e.text for e in row.find_elements(By.XPATH, ".//td")])
            link_element = row.find_element(By.XPATH, ".//a")
            link = link_element.get_attribute('href')

            # Open link in a new tab
            link_element.send_keys(Keys.CONTROL + Keys.RETURN)
            sleep(5)
            driver.switch_to.window(driver.window_handles[-1])

            # Try to get the PDF link
            pdf_links = [link.get_attribute('href') for link in driver.find_elements(By.TAG_NAME, 'a') if
                         link.get_attribute('href') and link.get_attribute('href').endswith('.pdf')]
            if pdf_links:
                links_metadata.append((pdf_links[0], metadata))

            # Close the tab and switch back to the main window
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
        return links_metadata

    def get_pdf_links_from_page(self, url: str, next_button_xpath: str = None, table_xpath: str = None) -> list:
        """Retrieve all PDF links from a webpage using undetected_chromedriver."""
        options = uc.ChromeOptions()
        with uc.Chrome(options=options) as driver:
            driver.get(url)
            pdf_links = []
            page_number = 1

            links_file = self.save_directory / "links.pkl"
            if links_file.exists():
                pdf_links, page_number = self.load_links_from_pickle(links_file)
                self.navigate_to_page(driver, next_button_xpath, page_number)

            logger.info("Starting to scrape PDF links, waiting 15 seconds for user to change links per page etc...")
            sleep(15)
            while True:
                sleep(8)
                if table_xpath:
                    links_metadata = self.get_gsa_links_and_metadata(driver, table_xpath)
                    for link, metadata in links_metadata:
                        pdf_links.append(link)
                        logger.info(f"Found link: {link} with metadata: {metadata}")
                else:
                    pdf_links.extend([link.get_attribute('href') for link in driver.find_elements(By.TAG_NAME, 'a') if
                                      link.get_attribute('href') and link.get_attribute('href').endswith('.pdf')])

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
        """Save the links and current page number."""
        with open(links_file, "wb") as f:
            pickle.dump({'links': pdf_links, 'page_number': page_number}, f)
            logger.info(f"Saved {len(pdf_links)} links to {links_file}")

    def process_organization(self, organization_info: dict) -> None:
        """Main function to download all PDFs from a given webpage."""
        for url in organization_info["urls"]:
            pdf_links = self.get_pdf_links_from_page(url, organization_info.get("xpath"),
                                                     organization_info.get("table_xpath"))
            for link in pdf_links:
                file_name = Path(link).name
                self.download_pdf(link, self.save_directory / file_name)
                logger.info(f"Downloaded {file_name}")
        logger.info(f"Downloaded {len(pdf_links)} PDFs to {self.save_directory}")


if __name__ == "__main__":
    organizations = {
        "GSA": {
            "urls": ["https://www.gsa.gov/forms", "https://www.gsa.gov/forms/obsolete-forms"],
            "xpath": "//a[@id='DataTables_Table_0_next']",
            "table_xpath": "//table[@id='DataTables_Table_0']/tbody/tr"
        }
    }

    for name, info in organizations.items():
        save_directory = Path(f"G:/s3/forms/PDF/{name}")
        save_directory.mkdir(parents=True, exist_ok=True)
        downloader = PDFDownloader(save_directory)
        downloader.process_organization(info)
