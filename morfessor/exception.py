class MorfessorException(Exception):
    """Base class for exceptions in this module."""
    pass


class ArgumentException(Exception):
    pass


class InvalidCategoryError(Exception):
    def __init__(self, category):
        Exception.__init__(
            self,
            u'This model does not recognize the category {}'.format(
                category))


class InvalidOperationError(Exception):
    def __init__(self, operation, function_name):
        Exception.__init__(
            self,
            (u'This model does not have a method ' +
             u'{}, and therefore cannot perform operation "{}"'.format(
                function_name, operation)))


class UnsupportedConfigurationError(MorfessorException):
    def __init__(self, reason):
        Exception.__init__(
            self,
            u'This operation is not supported in this program configuration. '
            u'Reason: {}.'.format(reason))
