# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import threading
import queue


class Worker(threading.Thread):
  """Thread executing tasks from a given task `Queue`.

  Args:
    tasks:
      Queue of tasks to consume be consumed.

  """
  def __init__(self, tasks):
    threading.Thread.__init__(self)
    self.tasks = tasks
    self.daemon = True
    self.results = []
    self.start()

  def run(self):
    """Run a task on the queue and append what is returned from the task to the `self.results`."""
    while True:
      func, args, kargs = self.tasks.get()
      try:
        self.results.append(func(*args, **kargs))
      except Exception as e:
        raise ValueError(str(e))
      finally:
        self.tasks.task_done()


class ThreadPool(object):
  """Pool of threads consuming tasks from a queue

  Args:
    num_threads:
      max number of threads to allow in the pool.

  """
  def __init__(self, num_threads):
    self.tasks = queue.Queue(num_threads)
    self.workers = []
    self.lock = threading.Lock()
    for _ in range(num_threads):
      self.workers.append(Worker(self.tasks))

  def add_task(self, func, *args, **kargs):
    """
    Add a task to the queue

    Args:
      func:
        task function to be called with `*args` and `**kargs`

    """
    self.tasks.put((func, args, kargs))

  def wait_completion(self):
    """Wait for completion of all the tasks in the queue.

    Returns:
      a list of lists containing returns from the task function.

    """
    self.tasks.join()
    results = [worker.results for worker in self.workers]
    return results
