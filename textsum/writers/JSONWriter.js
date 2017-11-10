'use strict';
const Writer = require('./Writer');
const jsonfile = require('jsonfile');
const path = require('path');

module.exports = class JSONWriter extends Writer {
  constructor(dst, options = {}) {
    super(dst, options);
    this.ext = options.ext || '.json';
  }

  write(name, data) {
    return new Promise((resolve, reject) => {
      const file = path.join(this.dst, `${name}${this.ext}`)
      jsonfile.readFile(file, (err, obj) => {
        const out = data;
        if (!err) {
          out.concat(data);
        }
        jsonfile.writeFile(file, out, (err) => {
          if (err) {
            return reject();
          }
          return resolve();
        });
      });
    })
  }
}
