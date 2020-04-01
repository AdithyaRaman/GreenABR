import os
from selenium import webdriver
from pyvirtualdisplay import Display
display = Display(visible=0, size=(800, 600))
display.start()
driver = webdriver.Chrome('../abr_browser_dir/driver/chromedriver')
driver.get("https://localhost/myindex_RL.html")
print ('haha')
driver.quit()
display.stop()