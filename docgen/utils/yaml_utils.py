from easydict import EasyDict as edict
import typing
import yaml
class MySafeYAMLDumper(yaml.Dumper):
    """
    Custom YAML dumper that only dumps the objects that can be read in later by yaml.safe_load.
    This version attempts to serialize objects using yaml.safe_dump() and falls back to a string
    representation if it fails.
    """

    def represent_data(self, data):
        # Check if the data is safe to dump by trying to dump it with safe_dump.
        try:
            yaml.safe_dump(data)
            # If no exception is raised, use the default representation method.
            return super().represent_data(data)
        except yaml.YAMLError:
            # If safe_dump fails, fall back to a string representation.
            return super().represent_data(str(data))


class ReallySafeDumper(MySafeYAMLDumper):
    """
    Custom YAML dumper that only dumps the objects that can be read in later by yaml.safe_load.
    This version always uses a string representation.
    """

    def represent_data(self, data):
        # Check if the data is a supported type, otherwise convert to string.
        if not isinstance(data, (str, int, float, list, tuple, dict)):
            data = str(data)
        # Use the default method to represent the data, which will handle recursion for complex types.
        return super().represent_data(data)


class Tagged(typing.NamedTuple):
    tag: str
    value: object


class NoAliasDumper(yaml.SafeDumper):
    """ Removes the *id002 memory references when lists/tuples are used multiple times
    """
    def ignore_aliases(self, data):
        return True

    def represent_tagged(self, data):
        assert isinstance(data, Tagged), data
        node = self.represent_data(data.__wrapped__)
        node.tag = data.tag
        return node
