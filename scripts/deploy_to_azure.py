"""Script for deploying the pneumonia classification model to Azure Container Instances using Azure SDK."""

import argparse
import logging
from typing import Any, Dict, Optional

from azure.identity import ClientSecretCredential, DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.containerinstance.models import (
    Container,
    ContainerGroup,
    ContainerPort,
    ImageRegistryCredential,
    IpAddress,
    OperatingSystemTypes,
    Port,
    ResourceRequests,
    ResourceRequirements,
)
from azure.mgmt.resource import ResourceManagementClient


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_credentials(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> Any:
    """
    Get Azure credentials for authentication.

    Args:
        client_id: Azure service principal client ID
        client_secret: Azure service principal client secret
        tenant_id: Azure tenant ID

    Returns:
        Azure credential object for authentication
    """
    if all([client_id, client_secret, tenant_id]):
        logger.info("Using service principal authentication")
        return ClientSecretCredential(
            tenant_id=tenant_id, client_id=client_id, client_secret=client_secret
        )
    else:
        logger.info("Using default Azure authentication")
        return DefaultAzureCredential()


def ensure_resource_group_exists(
    credential: Any, subscription_id: str, resource_group: str, location: str
) -> None:
    """
    Ensure that the resource group exists, creating it if necessary.

    Args:
        credential: Azure authentication credential
        subscription_id: Azure subscription ID
        resource_group: Name of the resource group
        location: Azure region location
    """
    resource_client = ResourceManagementClient(credential, subscription_id)

    if resource_group in [rg.name for rg in resource_client.resource_groups.list()]:
        logger.info(f"Resource group '{resource_group}' already exists")
    else:
        logger.info(f"Creating resource group '{resource_group}' in {location}")
        resource_client.resource_groups.create_or_update(
            resource_group, {"location": location}
        )


def deploy_to_azure(
    subscription_id: str,
    resource_group: str,
    location: str,
    registry_name: str,
    image_name: str,
    container_name: str,
    dns_name: str,
    cpu_cores: float = 1.0,
    memory_gb: float = 1.5,
    registry_username: Optional[str] = None,
    registry_password: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deploy the pneumonia classification model to Azure Container Instances.

    Args:
        subscription_id: Azure subscription ID
        resource_group: Name of the resource group
        location: Azure region (e.g., "eastus")
        registry_name: Name of the Azure Container Registry
        image_name: Name of the Docker image (e.g., "pneumonia-api:v1")
        container_name: Name for the container instance
        dns_name: DNS label name for the container
        cpu_cores: Number of CPU cores to allocate
        memory_gb: Amount of memory to allocate in GB
        registry_username: ACR username (if needed)
        registry_password: ACR password (if needed)
        client_id: Azure service principal client ID
        client_secret: Azure service principal client secret
        tenant_id: Azure tenant ID

    Returns:
        Dictionary with deployment results including FQDN
    """
    # Get Azure credentials
    credential = get_credentials(client_id, client_secret, tenant_id)

    # Ensure resource group exists
    ensure_resource_group_exists(credential, subscription_id, resource_group, location)

    # Full image name with registry
    if not registry_username:
        registry_username = registry_name

    full_image_name = f"{registry_name}.azurecr.io/{image_name}"
    logger.info(f"Deploying image: {full_image_name}")

    # Create container instance client
    aci_client = ContainerInstanceManagementClient(credential, subscription_id)

    # Configure the container
    container_resource = Container(
        name=container_name,
        image=full_image_name,
        resources=ResourceRequirements(
            requests=ResourceRequests(memory_in_gb=memory_gb, cpu=cpu_cores)
        ),
        ports=[ContainerPort(port=8000)],
    )

    # Create container group with public IP
    logger.info(f"Creating container group: {container_name}")
    container_group = ContainerGroup(
        location=location,
        containers=[container_resource],
        os_type=OperatingSystemTypes.LINUX,
        restart_policy="Always",
        ip_address=IpAddress(
            type="Public",
            ports=[Port(port=8000, protocol="TCP")],
            dns_name_label=dns_name,
        ),
        image_registry_credentials=[
            ImageRegistryCredential(
                server=f"{registry_name}.azurecr.io",
                username=registry_username,
                password=registry_password,
            )
        ]
        if registry_password
        else None,
    )

    # Deploy the container group
    aci_client.container_groups.begin_create_or_update(
        resource_group, container_name, container_group
    ).result()

    # Get deployment details
    container_details = aci_client.container_groups.get(resource_group, container_name)
    fqdn = container_details.ip_address.fqdn
    ip_address = container_details.ip_address.ip

    logger.info(f"Deployment completed: {container_name}")
    logger.info(f"FQDN: {fqdn}")
    logger.info(f"IP: {ip_address}")
    logger.info(f"API URL: http://{fqdn}:8000")

    return {
        "name": container_name,
        "fqdn": fqdn,
        "ip": ip_address,
        "status": container_details.provisioning_state,
        "api_url": f"http://{fqdn}:8000",
    }


def main() -> None:
    """Parse command line arguments and deploy to Azure."""
    parser = argparse.ArgumentParser(
        description="Deploy pneumonia classification model to Azure"
    )

    # Required arguments
    parser.add_argument(
        "--subscription-id", required=True, type=str, help="Azure subscription ID"
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--resource-group",
        type=str,
        default="pneumonia-classifier-rg",
        help="Azure resource group name",
    )
    parser.add_argument(
        "--location", type=str, default="eastus", help="Azure region location"
    )
    parser.add_argument(
        "--registry-name",
        type=str,
        default="pneumoniaclassifieracr",
        help="Azure Container Registry name",
    )
    parser.add_argument(
        "--image-name",
        type=str,
        default="pneumonia-api:v1",
        help="Docker image name and tag",
    )
    parser.add_argument(
        "--container-name",
        type=str,
        default="pneumonia-api",
        help="Container instance name",
    )
    parser.add_argument(
        "--dns-name",
        type=str,
        default="pneumonia-classifier",
        help="DNS name label for the container",
    )
    parser.add_argument("--cpu", type=float, default=1.0, help="Number of CPU cores")
    parser.add_argument("--memory", type=float, default=1.5, help="Memory in GB")

    # Authentication options
    parser.add_argument(
        "--registry-username", type=str, help="ACR username (defaults to registry name)"
    )
    parser.add_argument(
        "--registry-password", type=str, help="ACR password (required if using ACR)"
    )
    parser.add_argument(
        "--client-id",
        type=str,
        help="Azure service principal client ID for authentication",
    )
    parser.add_argument(
        "--client-secret",
        type=str,
        help="Azure service principal client secret for authentication",
    )
    parser.add_argument(
        "--tenant-id", type=str, help="Azure tenant ID for authentication"
    )

    args = parser.parse_args()

    # Deploy to Azure
    _ = deploy_to_azure(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        location=args.location,
        registry_name=args.registry_name,
        image_name=args.image_name,
        container_name=args.container_name,
        dns_name=args.dns_name,
        cpu_cores=args.cpu,
        memory_gb=args.memory,
        registry_username=args.registry_username,
        registry_password=args.registry_password,
        client_id=args.client_id,
        client_secret=args.client_secret,
        tenant_id=args.tenant_id,
    )

    logger.info("Deployment completed successfully and result handled internally")


if __name__ == "__main__":
    main()
