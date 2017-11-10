'use strict';
const writer = require('../writers');
const Scraper = require('./Scraper');
const puppeteer = require('puppeteer');

module.exports = class Medium extends Scraper {
  constructor(options = {}) {
    super(options);
    this._baseUrl = 'https://medium.com';
    this._topicsUrl = `${this._baseUrl}/topics`;
    this._writerCtor = writer.lookup(options.fmt || 'json')
  }

  /**
   * fetchTopics scrapes all topics from the medium page.
   * @return {Array}
   */
  async _fetchTopics(browser) {
    const page = await browser.newPage();
    await page.goto(this._topicsUrl, {timeout: 30000});
    const topics = await page.evaluate(() => {
      const topicObjs = document.querySelectorAll('[data-index]');
      return Object.keys(topicObjs)
        .map((key) => {
          const topic = topicObjs[key];
          return {
            href: topic.lastChild.getAttribute('href'),
            name: topic.textContent,
          };
        });
    });
    await page.close();
    this._logger.log(`Fetched topics: "${topics.map(t => t.name).join('", "')}".`);
    return topics;
  }

  /**
   * fetchArticles scrapes all articles from the medium topic page.
   * @param  {string} topicUrl where articles are located based on topic.
   * @return {Array}
   */
  async _fetchArticles(browser, topicUrl) {
    if (topicUrl === `${this._baseUrl}/`) {
      this._logger.log(`√ Skipping ${topicUrl} for now: not implemented yet.`);
      return [];
    }
    let page;
    try {
      page = await browser.newPage();
      this._logger.log(`- Waiting for ${topicUrl}...`);
      await page.goto(topicUrl, {timeout: 30000});
    } catch(error) {
      this._logger.error(`X ${topicUrl} failed with error: ${error.message}`);
      return [];
    }

    this._logger.log(`- Scrapping articles for ${topicUrl}...`);
    // extract article links from main page.
    let articleLinks = await page.evaluate(() => {
      const articleObjs = document.querySelectorAll('[data-post-id]');
      return Object.keys(articleObjs).map((key) => {
        return articleObjs[key].getAttribute('href');
      });
    });

    if (!articleLinks) {
      await page.close();
      this._logger.warn(`W ${topicUrl} recieved 0 articles`);
      return [];
    }

    this._logger.log(`√ ${topicUrl} recieved ${articleLinks.length} articles`);
    this._logger.log('- Precaching pages...');

    articleLinks = articleLinks.filter(a => !!a);

    let cachedPages = [];
    for (let i = 0; i < articleLinks.length; i++) {
      cachedPages.push({page: await browser.newPage(), href: articleLinks[i]});
    }

    this._logger.log(`- Gathering ${topicUrl} data for articles...`);
    const articleData = await Promise.all(cachedPages.map(async ({page, href}) => {
      try {
        this._logger.log(`- Waiting for ${href}...`);
        await page.goto(href, {timeout: 30000});
      } catch(error) {
        this._logger.error(`X ${href} failed with error: ${error.message}`);
        return undefined;
      }
      const res = await page.evaluate(() => (
        document.getElementsByTagName('script')['8'].innerHTML
      ));
      await page.close();
      this._logger.log(`√ Scraped ${href}`);
      return res;
    }));

    // return data as is from the medium site rendered object.
    return articleData.filter((a) => !!a).map((str) => {
      if (!str.trim) {
        return undefined;
      }
      const articleObjRegex = /(?:window\[\"obvInit\"\]\()(\{.*?[\}\)])(?:\))/;
      const stripped = str.trim().match(articleObjRegex)
      let parsed = null;
      try {
        parsed = JSON.parse(stripped[1]);
      } catch(e) {
        parsed = undefined;
      }
      return parsed;
    }).filter((a) => !!a);
  }

  _parseArticles(articles, topic) {
    return articles.map(({value}) => {
      const {virtuals, content, title} = value;
      const res = {
        title,
        topic,
        tags: virtuals.tags.map(({slug}) => slug),
        subtitle: content.subtitle,
        text: content.bodyModel.paragraphs.reduce((pre, {text}) => {
          return [pre, text].join('\n');
        }, ''),
      };
      if (value.homeCollection) {
        const col = value.homeCollection;
        if (col.description) {
          res.description = col.description;
        }
        if (col.shortDescription) {
          res.shortDescription = col.shortDescription;
        }
      }
      return res;
    });
  }

  /**
   * fetch crawls the Medium "corpus" feed by topic and writes to the provided directory.
   * @param  {string}  dir target destination to write to.
   * @return {Promise}
   */
  async fetch(dir) {
    const browser = await puppeteer.launch();
    const topics = await this._fetchTopics(browser);

    let writers = [];
    for (let i = 0; i < topics.length; i++) {
      let {href, name} = topics[i];
      const articles = await this._fetchArticles(browser, href);
      this._logger.log(`√ ${name} found ${articles.length} articles.`);
      const writer = new this._writerCtor(dir, {logger: this._logger});
      const data = this._parseArticles(articles, name)
      this._logger.log(`Writing ${name} with ${data.length} articles.`);
      writers.shift(writer.write(name, data));
    }
    await browser.close();
    return Promise.all(writers);
  }
};
