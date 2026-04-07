class DeploymentNotFoundError(Exception):
    """
    Raised when a deployment is not found
    """

    pass


class ReplicaSetNotFoundError(Exception):
    """
    May be raised if a deployment does not set have a replica set
    """

    pass
