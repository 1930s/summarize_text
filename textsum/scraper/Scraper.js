'use strict';

module.exports = class Scraper {
  constructor(options) {
    this._logger = options.logger || console;
  }

  fetch() {
    throw Error('fetch not implemented');
  }
};
