import scrapy
from scrapy.crawler import CrawlerProcess


class LiquipediaStatsSpider(scrapy.Spider):
    name = "liquipedia_stats"

    start_urls = [
        "https://liquipedia.net/dota2/BLAST/Slam/4/Statistics",
    ]

    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "USER_AGENT": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/129.0.0.0 Safari/537.36"
        ),
        "FEEDS": {
            "blast_slam_4_stats.csv": {
                "format": "csv",
                "encoding": "utf-8",
                "overwrite": True,
            }
        },
    }

    def parse(self, response):
        for row in response.css("tr.dota-stat-row"):
            cells = row.css("td")

            cell_values = [
                " ".join(td.css("::text").getall()).strip()
                for td in cells
            ]

            hero = None
            if len(cells) > 1:
                hero_text_parts = cells[1].css("::text").getall()
                hero = " ".join(t.strip() for t in hero_text_parts if t.strip()) or None

            item = {}

            if hero:
                item["hero"] = hero

            for idx, value in enumerate(cell_values):
                item[f"col_{idx}"] = value

            if any(v for v in item.values()):
                yield item


def run_spider():
    process = CrawlerProcess()
    process.crawl(LiquipediaStatsSpider)
    process.start()


if __name__ == "__main__":
    run_spider()


