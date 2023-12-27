class AddressReuse(Exception):
    """Attempt to re-write an address in a GenJAX trace. Any given
    address for a random choice may only be written to once. You can
    choose a different name for the choice, or nest it into a scope
    where it is unique."""


class StaticAddressJAX(Exception):
    """Static addresses must not contain JAX traced values"""
