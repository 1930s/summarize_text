# running

**Before** you start slinging, do this first (> Node 7, > Python 2 or 3):

```sh
npm i && python setup.py install
```

1. Scrape dataset `medium`. This will go for a while dependent on your internet connection.
   There is a timeout of 30s where the page will be skipped. This tries to use 8 threads.

   ```sh
   bin/scrape medium
   ```

   This crawls the [topics page](https://medium.com/topics) and collects the available topics. Then
   it visits each topic main page i.e. [culture](https://medium.com/topics/culture) and extracts all
   the landing page articles (extracts the `href` according to the attribute `data-post-id`). Finally,
   it visits each article page, finds the Medium API from the landing page, it looks like this:

   ```html
   <script>
   // <![CDATA[
   window["obvInit"]({"value":{...}});
   </script>
   ```

   It uses `page.evaluate(...)` to perform a `regexp` on the script content, parses it as JSON and
   then passes it back to node.js. It finally strips the meta data, it reduces the object as a model
   for the python `textsum/dataset/article.py` model `textsum.Article` with the features: `title`,
   `subtitle`, `text`, `tags`, `description`, `short_description`.

   We now have raw data that we can use to do fun things.

2. Convert raw data to numpy records of examples.

   ```sh
   bin/records --src=data/medium --dst=records/medium --pad
   ```

   This takes the raw data from `src` and serializes it as `textsum.Article` objects for consumption.
   As it is serializing, it tokenizes all the features (`title`, `subtitle`, ...) as mentioned in **2**.
   It saves all these as `np.ndarray`s and stores them in `dst` by `topic`. Next, the examples
   are piped to `*.npy` files. This comes in handy to be used with the
   native `tf.data` API, it's like **hadoop** or **spark** but native compatibility with **tensorflow**.
   Finally, all the record `tokens` we collected for each topic, is collected in a `set`, so we don't
   store all tokens in memory to avoid repetition, this is done in a `map->reduce` fashion. The tokens
   are gathered by `topic` on a individual thread as a `set` of `str`s and the `union` operation reduces
   the total space for each `topic` `map` operation. The `map` stage returns all the individual vocabs
   for each feature (as in **2**) and is reduced by the `union` operation again.

   We now have a set of vocab files for each feature in the dataset.

3. Final step, **Sling** (run the experiment)

  ```sh
  bin/experiment \
    --model_dir=article_model \
    --dataset_dir=records/medium \
    --input_feature='text' \
    --target_feature='title' \
    --schedule='train'
  ```
