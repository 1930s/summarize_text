const Writer = require('./Writer');
const JSONWriter = require('./JSONWriter');

module.exports = {
  Writer,
  JSONWriter,
  lookup(type) {
    switch (type) {
      case 'json':
      return JSONWriter;
      default:
      throw new TypeError(`${type} not supported.`);
    }
  }
};
