from playwright.sync_api import sync_playwright
import os

os.makedirs("images", exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=False) #browser = p.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()

    image_urls = []

    def handle_response(response):
        if "get_image.php" in response.url:
            if "image" in response.headers.get("content-type", ""):
                image_urls.append(response.url)

    page.on("response", handle_response)

    url = "https://csbs.shogakukan.co.jp/bookshelf"
    page.goto(url)#"https://pages.csbs.shogakukan.co.jp")

    input("Log in and navigate to the content, then press Enter...") # pauses program until user sets the page up

    # wait for content to fully load
    page.wait_for_timeout(120000) # milliseconds - 120 seconds

    # download images while session is valid
    for i, url in enumerate(image_urls):
        r = context.request.get(url)
        with open(f"images/page_{i}.jpg", "wb") as f:
            f.write(r.body())
    browser.close()

print(f"Saved {len(image_urls)} images.")
