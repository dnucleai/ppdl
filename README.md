# PPDL

## Install requirements

Run ``pip install -r requirements.txt`` to install the requirements for this project.

## Test

We have provided a test implementation of the PPDL protocol using 2 clients running a simple CNN model on the MNIST dataset.

To run this test, execute

```
bin/run-test python ppdl/test/simple/simple_server.py
```

In another window, execute

```
bin/run-test python ppdl/test/simple/simple_client.py --train 0
```

In a third window, execute

```
bin/run-test python ppdl/test/simple/simple_client.py --train 1
```
