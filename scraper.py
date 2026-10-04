import requests
from bs4 import BeautifulSoup
import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import csv
from datetime import datetime, timedelta

class ExpediaFlightScraper:
    def __init__(self):
        """Initialize the scraper with Selenium WebDriver"""
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        self.wait = WebDriverWait(self.driver, 10)
        
    def get_us_airports(self):
        """Get list of major US airports to scan"""
        # This is a simplified list - you'd need a comprehensive database
        us_airports = [
            "JFK", "LAX", "ORD", "DFW", "DEN", "ATL", "SFO", "SEA", 
            "LAS", "MCO", "EWR", "MIA", "BOS", "PHX", "IAH", "CLT"
        ]
        return us_airports
    
    def search_flights(self, origin, destination, date):
        """Search for flights between two airports on a specific date"""
        try:
            # Construct Expedia URL
            url = f"https://www.expedia.com/Flights-Search?flight-type=on&mode=search&trip=oneway&leg1=from:{origin},to:{destination},departure:{date}TANYT&passengers=adults:1&options=cabinclass:&stopNumber=nonstop"
            
            self.driver.get(url)
            time.sleep(random.uniform(3, 5))  # Random delay to appear human
            
            # Wait for results to load
            try:
                self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-test-id='flight-card']")))
            except TimeoutException:
                print(f"No results for {origin} to {destination}")
                return []
            
            flights = []
            flight_cards = self.driver.find_elements(By.CSS_SELECTOR, "[data-test-id='flight-card']")
            
            for card in flight_cards:
                try:
                    # Extract flight information - these selectors will need updating
                    airline_element = card.find_element(By.CSS_SELECTOR, "[data-test-id='airline-name']")
                    airline = airline_element.text if airline_element else "N/A"
                    
                    # Parse airline code from text (this is simplified)
                    airline_code = airline.split()[-1] if " " in airline else airline[:2]
                    
                    times = card.find_elements(By.CSS_SELECTOR, "[data-test-id='departure-arrival-times']")
                    if len(times) >= 2:
                        departure_time = times[0].text
                        arrival_time = times[1].text
                    else:
                        departure_time = arrival_time = "N/A"
                    
                    price_element = card.find_element(By.CSS_SELECTOR, "[data-test-id='price']")
                    total_fare = price_element.text if price_element else "N/A"
                    
                    flight_data = {
                        'origin': origin,
                        'dest': destination,
                        'totalFare': total_fare,
                        'departure_time': departure_time,
                        'arrival_time': arrival_time,
                        'airline': airline,
                        'airlineCode': airline_code
                    }
                    
                    flights.append(flight_data)
                    
                except NoSuchElementException as e:
                    print(f"Error parsing flight card: {e}")
                    continue
            
            return flights
            
        except Exception as e:
            print(f"Error searching {origin} to {destination}: {e}")
            return []
    
    def scrape_all_domestic_flights(self):
        """Scrape all domestic US nonstop flights"""
        airports = self.get_us_airports()
        all_flights = []
        
        # Use a date in the future (e.g., 30 days from now)
        future_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        
        # Scan all airport pairs
        for i, origin in enumerate(airports):
            for dest in airports[i+1:]:  # Avoid duplicates
                print(f"Scanning {origin} to {dest} on {future_date}")
                
                flights = self.search_flights(origin, dest, future_date)
                all_flights.extend(flights)
                
                # Save to CSV periodically
                self.save_to_csv(all_flights, "expedia_flights.csv")
                
                # Random delay between searches
                time.sleep(random.uniform(5, 10))
        
        return all_flights
    
    def save_to_csv(self, flights, filename):
        """Save flight data to CSV file"""
        if not flights:
            return
        
        keys = ['origin', 'dest', 'totalFare', 'departure_time', 'arrival_time', 'airline', 'airlineCode']
        
        with open(filename, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(flights)
        
        print(f"Saved {len(flights)} flights to {filename}")
    
    def close(self):
        """Close the WebDriver"""
        self.driver.quit()

# Usage example
if __name__ == "__main__":
    scraper = ExpediaFlightScraper()
    
    try:
        # For testing, use a smaller subset of airports
        # In production, you'd use all airports
        all_flights = scraper.scrape_all_domestic_flights()
        print(f"Total flights scraped: {len(all_flights)}")
        
    except Exception as e:
        print(f"Error during scraping: {e}")
    
    finally:
        scraper.close()