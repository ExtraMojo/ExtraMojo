"""
Reservoir sampling on a stream.

# References
- Algorithm R: https://en.wikipedia.org/wiki/Reservoir_sampling
"""

from random import random_ui64


struct ReservoirSampler[T: Copyable & Movable](
    Copyable, Movable, ExplicitlyCopyable
):
    """Sample N items from a stream of unknown length.

    Sample all the elements, this should retain the order since we always automatically take the first N elements.

    ```mojo
    from random import seed
    from testing import assert_equal

    from ExtraMojo.stats.sampling import ReservoirSampler

    # Set the global random seed
    seed(42)

    var sampler = ReservoirSampler[Int](10)

    var items = List(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    for item in items:
        sampler.insert(item)
    assert_equal(sampler.reservoir, items)
    ```

    Sample only a subset of the input list.

    ```mojo
    var sampler = ReservoirSampler[Int](5)

    for item in items:
        sampler.insert(item)
    assert_equal(len(sampler.reservoir), 5)
    assert_equal(sampler.reservoir, List(0, 9, 2, 3, 7))
    ```

    Sample only a single element.

    ```mojo
    var sampler = ReservoirSampler[Int](1)

    for item in items:
        sampler.insert(item)
    assert_equal(len(sampler.reservoir), 1)
    assert_equal(sampler.reservoir, List(6))
    ```

    Sample more elements than are in the input stream.
    ```mojo
    var sampler = ReservoirSampler[Int](11)

    for item in items:
        sampler.insert(item)
    assert_equal(len(sampler.reservoir), 10)
    assert_equal(sampler.reservoir, items)
    ```

    Sample zero elements
    ```mojo
    var sampler = ReservoirSampler[Int](0)

    for item in items:
        sampler.insert(item)
    assert_equal(len(sampler.reservoir), 0)
    assert_equal(sampler.reservoir, List[Int]())
    ```
    """

    var reservoir: List[T]
    var values_to_collect: Int
    var seen_values: Int

    fn __init__(out self, values_to_collect: Int):
        self.seen_values = 0
        self.reservoir = List[T](capacity=values_to_collect)
        self.values_to_collect = values_to_collect

    fn insert(mut self, read item: T):
        """Add an element.

        The element will be tested for addition to the reservoir.
        """
        if len(self.reservoir) < self.values_to_collect:
            self.reservoir.append(item)
            self.seen_values += 1
            return

        var index = random_ui64(0, self.seen_values)
        if index < self.values_to_collect:
            self.reservoir[Int(index)] = item
        self.seen_values += 1
