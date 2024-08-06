from selenium import webdriver
import os
import time

def capture_image(Map, file_name,location):
    mapFname = f'capture/{location}/output.html'
    Map.save(mapFname)
    
    mapUrl = 'file://{0}/{1}'.format(os.getcwd(), mapFname)
    driver = webdriver.Firefox()
    driver.get(mapUrl)
    
    # wait for 5 seconds for the maps and other assets to be loaded in the browser
    time.sleep(5)
    driver.save_screenshot(f'capture/{location}/'+file_name+'.png')
    driver.quit()