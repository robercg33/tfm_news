import pandas as pd
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# Function to configure session with retry strategy
def create_session():
    session = requests.Session()
    retry = Retry(
        total=1,  # Total number of retries
        backoff_factor=1,  # A backoff factor to apply between attempts
        status_forcelist=[500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["HEAD", "GET", "OPTIONS"]  # Retry only these methods
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# Function to scrape a single page
def scrape_page(url, session, timeout=5):
    """
    Scrapes the content of a single web page and returns its title and text.

    Args:
        url (str): The URL of the web page to scrape.
        session (requests.Session): The requests session object to use for making the HTTP request.
        timeout (int, optional): The timeout value for the HTTP request in seconds. Default is 5.

    Returns:
        dict: A dictionary with the URL as the key and a list containing the title and the concatenated text of all paragraphs as the value. If an error occurs during the request, the value will be None.

    Raises:
        requests.RequestException: If an error occurs during the HTTP request.

    Example:
        session = requests.Session()
        result = scrape_page('http://example.com', session)
        print(result)
        # Output: {'http://example.com': ['Example Domain', 'This domain is for use in illustrative examples ...']}
    """

    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()  # Raise HTTPError for bad responses

        #Parse the text with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        #Get the paragraphs
        paragraphs = soup.find_all('p')

        #Get the raw text of the paragraph
        res_list = [elem.get_text(strip=True) for elem in paragraphs]

        #Get the title
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text()
        else:
            title = None
        
        #Return the joined text
        return {url: [title, ". ".join(res_list)]}
    except requests.RequestException as e:
        #print(f"Error scraping {url}: {e}")
        return {url: None}



# Function to handle parallel scraping
def parallel_scraping(urls, max_workers=5, timeout=5):
    """
    Handles the parallel scraping of multiple web pages using a thread pool.

    Args:
        urls (list of str): A list of URLs to be scraped.
        max_workers (int, optional): The maximum number of threads to use for parallel scraping. Default is 5.
        timeout (int, optional): The timeout value for each HTTP request in seconds. Default is 5.

    Returns:
        list of dict: A list of dictionaries containing the scraped data. Each dictionary has the URL as the key and a list containing the title and the concatenated text of all paragraphs as the value. If an error occurs during the request for a URL, the value will be None.

    Example:
        urls = ['http://example.com', 'http://example.org']
        results = parallel_scraping(urls, max_workers=10)
        print(results)
        # Output: [{'http://example.com': ['Example Domain', 'This domain is for use in illustrative examples ...']}, ...]

    Notes:
        This function uses a thread pool to execute the scraping tasks concurrently. The `tqdm` library is used to display a progress bar.

    """
    #Create the list to store the results and a session
    results = []
    session = create_session()

    #Create a ThreadPoolExecutor to manage the pool of worker threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        #Submit scraping tasks to the executor and map futures to URLs
        future_to_url = {executor.submit(scrape_page, url, session, timeout): url for url in urls}

        #Process the futures as they complete, showing progress with tqdm
        for future in tqdm(as_completed(future_to_url), total=len(urls), desc="Scraping progress"):
            try:
                #Retrieve the result of the future (scraped data)
                data = future.result()

                #Add to the results lists
                results.append(data)
            except Exception as e:

                #If an exception occurs, print the exception details
                print(f"An exception ocurred: {e}")
                
    #Print completion message and return the resulting list
    print("Scraped completed!")
    return results


#Main function
def lambda_handler(event, context):

    #Get the list of urls
    urls = event["urls"]

    #The number of max parallel workers, 10 by default
    max_workers = event.get("max_workers", 10)

    #The defined timeout, 5 by default
    timeout = event.get("timeout", 5)

    #Execute and get the results
    results = parallel_scraping(urls, max_workers=max_workers, timeout=timeout)

    #Create a dataframe excluding the non-null elements
    results_df = pd.DataFrame([{"url": k, "title": v[0], "body": v[1]} for d in results for k, v in d.items() if v is not None])

    #Return the results in json format
    return results_df.to_json(orient="records")
