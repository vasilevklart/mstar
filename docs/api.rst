API Reference
=============

Generated from the docstrings in the subpackages listed below. ``mstar.client`` is the Python
SDK (``from mstar import MStarClient``) and ``mstar.cli`` is the ``mstar`` command
(``mstar serve``); the rest are the server's internals.

.. note::

   The reference is built by importing each module. The docs job installs CPU-only torch,
   the base package (``pip install -e .``) and ``docs/requirements.txt``
   (``.github/workflows/docs.yml``). A module that imports a package outside the base
   install at module scope, such as ``triton`` or ``transformers``, fails to import and gets
   no page.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   mstar.client
   mstar.cli
   mstar.api_server
   mstar.communication
   mstar.conductor
   mstar.engine
   mstar.graph
   mstar.model
   mstar.observability
   mstar.streaming
   mstar.utils
   mstar.worker
