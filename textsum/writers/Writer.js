'use strict';

module.exports = class Writer {
  constructor(dst, options = {}) {
    if (!dst) {
      throw new Error(`${this.name} requires a 'dst'.`);
    }
    this.dst = dst;
    this.logger = options.logger || console;
  }

  write(name, data) {
    throw new Error('write not implemented.');
  }
}
