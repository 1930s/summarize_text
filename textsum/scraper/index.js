'use strict';
const Scraper = require('./Scraper');
const Medium = require('./Medium');

module.exports = {
  Scraper,
  Medium,
  lookup(type) {
    switch (type) {
      case 'medium':
      return Medium;
      default:
      throw new TypeError(`${type} not supported.`);
    }
  }
};
