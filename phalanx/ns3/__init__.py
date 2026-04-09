"""ns-3 bridge interface for Phalanx.

Provides the API for integrating Phalanx schedulers with ns-3 via
the ns3-ai gym interface.  Full ns-3 integration is future work;
this module defines the stable API surface.
"""
from phalanx.ns3.bridge import NS3Bridge

__all__ = ["NS3Bridge"]
