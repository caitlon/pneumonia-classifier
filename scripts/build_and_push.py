"""Script for building and pushing Docker image to Azure Container Registry."""

import os
import argparse
import logging
import subprocess
from typing import Optional, List, Dict, Any

from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.mgmt.containerregistry import ContainerRegistryManagementClient
from azure.mgmt.resource import ResourceManagementClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(command: List[str]) -> str:
    """
    Run a shell command and return the output.
    
    Args:
        command: List of command parts to execute
        
    Returns:
        Command output as string
        
    Raises:
        RuntimeError: If command execution fails
    """
    logger.info(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        raise RuntimeError(f"Command failed: {e.stderr}")


def get_azure_credentials(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> Any:
    """
    Get Azure credentials for authentication.
    
    Args:
        client_id: Azure service principal client ID
        client_secret: Azure service principal client secret
        tenant_id: Azure tenant ID
        
    Returns:
        Azure credential object
    """
    if all([client_id, client_secret, tenant_id]):
        logger.info("Using service principal authentication")
        return ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )
    else:
        logger.info("Using default Azure authentication")
        return DefaultAzureCredential()


def ensure_resource_group_exists(
    credential: Any,
    subscription_id: str,
    resource_group: str,
    location: str
) -> None:
    """
    Ensure the resource group exists, creating it if necessary.
    
    Args:
        credential: Azure authentication credential
        subscription_id: Azure subscription ID
        resource_group: Resource group name
        location: Azure region location
    """
    resource_client = ResourceManagementClient(credential, subscription_id)
    
    if resource_group in [rg.name for rg in resource_client.resource_groups.list()]:
        logger.info(f"Resource group '{resource_group}' already exists")
    else:
        logger.info(f"Creating resource group '{resource_group}' in {location}")
        resource_client.resource_groups.create_or_update(
            resource_group,
            {"location": location}
        )


def ensure_acr_exists(
    credential: Any,
    subscription_id: str,
    resource_group: str,
    registry_name: str,
    location: str,
    sku: str = "Basic"
) -> Dict[str, Any]:
    """
    Ensure Azure Container Registry exists, creating it if necessary.
    
    Args:
        credential: Azure authentication credential
        subscription_id: Azure subscription ID
        resource_group: Resource group name
        registry_name: Container registry name
        location: Azure region location
        sku: Registry SKU (Basic, Standard, Premium)
        
    Returns:
        Dictionary with registry information
    """
    acr_client = ContainerRegistryManagementClient(credential, subscription_id)
    
    try:
        registry = acr_client.registries.get(resource_group, registry_name)
        logger.info(f"Container registry '{registry_name}' already exists")
        
        # Ensure admin user is enabled for password access
        if not registry.admin_user_enabled:
            logger.info("Enabling admin user for registry")
            registry = acr_client.registries.update(
                resource_group,
                registry_name,
                {"admin_user_enabled": True}
            )
    except:
        logger.info(f"Creating container registry '{registry_name}' in {location}")
        registry = acr_client.registries.begin_create(
            resource_group,
            registry_name,
            {
                "location": location,
                "sku": {"name": sku},
                "admin_user_enabled": True
            }
        ).result()
    
    # Get registry credentials
    credentials = acr_client.registries.list_credentials(resource_group, registry_name)
    
    return {
        "login_server": f"{registry_name}.azurecr.io",
        "username": credentials.username,
        "password": credentials.passwords[0].value,
        "resource_id": registry.id
    }


def build_and_push_image(
    subscription_id: str,
    resource_group: str,
    registry_name: str,
    image_name: str,
    location: str = "eastus",
    dockerfile_path: str = "docker/Dockerfile.api",
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build and push Docker image to Azure Container Registry.
    
    Args:
        subscription_id: Azure subscription ID
        resource_group: Resource group name
        registry_name: Container registry name
        image_name: Image name and tag (e.g., "pneumonia-api:v1")
        location: Azure region
        dockerfile_path: Path to Dockerfile
        client_id: Azure service principal client ID
        client_secret: Azure service principal client secret
        tenant_id: Azure tenant ID
        
    Returns:
        Dictionary with image information
    """
    # Get Azure credentials
    credential = get_azure_credentials(client_id, client_secret, tenant_id)
    
    # Ensure resource group exists
    ensure_resource_group_exists(credential, subscription_id, resource_group, location)
    
    # Ensure ACR exists and get credentials
    acr_info = ensure_acr_exists(
        credential, 
        subscription_id, 
        resource_group, 
        registry_name, 
        location
    )
    
    full_image_name = f"{acr_info['login_server']}/{image_name}"
    
    # Docker login to ACR
    logger.info(f"Logging in to ACR: {acr_info['login_server']}")
    run_command([
        "docker", "login", 
        acr_info['login_server'], 
        "--username", acr_info['username'], 
        "--password", acr_info['password']
    ])
    
    # Build Docker image
    logger.info(f"Building Docker image: {full_image_name}")
    run_command([
        "docker", "build", 
        "--platform", "linux/amd64",
        "-t", full_image_name, 
        "-f", dockerfile_path, 
        "."
    ])
    
    # Push image to ACR
    logger.info(f"Pushing image to ACR: {full_image_name}")
    run_command(["docker", "push", full_image_name])
    
    logger.info(f"Image successfully pushed: {full_image_name}")
    
    return {
        "registry": acr_info['login_server'],
        "image_name": image_name,
        "full_image_name": full_image_name,
        "username": acr_info['username'],
        "password": acr_info['password']
    }


def main() -> None:
    """Parse command line arguments and execute build and push."""
    parser = argparse.ArgumentParser(
        description="Build and push Docker image to Azure Container Registry"
    )
    
    # Required arguments
    parser.add_argument("--subscription-id", required=True, type=str,
                      help="Azure subscription ID")
    
    # Optional arguments with defaults
    parser.add_argument("--resource-group", type=str, default="pneumonia-classifier-rg",
                      help="Azure resource group name")
    parser.add_argument("--registry-name", type=str, default="pneumoniaclassifieracr",
                      help="Azure Container Registry name")
    parser.add_argument("--image-name", type=str, default="pneumonia-api:v1",
                      help="Docker image name and tag")
    parser.add_argument("--location", type=str, default="eastus",
                      help="Azure region location")
    parser.add_argument("--dockerfile", type=str, default="docker/Dockerfile.api",
                      help="Path to Dockerfile")
    
    # Authentication options
    parser.add_argument("--client-id", type=str,
                      help="Azure service principal client ID")
    parser.add_argument("--client-secret", type=str,
                      help="Azure service principal client secret")
    parser.add_argument("--tenant-id", type=str,
                      help="Azure tenant ID")
    
    args = parser.parse_args()
    
    # Build and push image
    result = build_and_push_image(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        registry_name=args.registry_name,
        image_name=args.image_name,
        location=args.location,
        dockerfile_path=args.dockerfile,
        client_id=args.client_id,
        client_secret=args.client_secret,
        tenant_id=args.tenant_id
    )
    
    logger.info("=== Image Build and Push Summary ===")
    logger.info(f"Registry: {result['registry']}")
    logger.info(f"Image: {result['image_name']}")
    logger.info(f"Full image name: {result['full_image_name']}")
    logger.info("")
    logger.info("To deploy this image, run:")
    logger.info(f"python scripts/deploy_to_azure.py --subscription-id {args.subscription_id} " +
               f"--registry-name {args.registry_name} --image-name {args.image_name} " +
               f"--registry-username {result['username']} --registry-password {result['password']}")


if __name__ == "__main__":
    main() 