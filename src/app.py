import asyncio
import os
import logging
import requests
from typing import Annotated, Dict, List, Optional
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.functions import kernel_function
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
# Set the logging level for  semantic_kernel.kernel to DEBUG.
logging.basicConfig(
    format="[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("kernel").setLevel(logging.DEBUG)

class AzureDocsPlugin:
    """Plugin for providing information about Azure services and documentation."""
    
    # Core Azure service categories and their key services
    azure_services = {
        "Compute": [
            {"name": "Virtual Machines", "description": "Provision Windows and Linux virtual machines in seconds", "doc_url": "https://docs.microsoft.com/azure/virtual-machines/"},
            {"name": "App Service", "description": "Quickly create powerful cloud apps for web and mobile", "doc_url": "https://docs.microsoft.com/azure/app-service/"},
            {"name": "Azure Functions", "description": "Process events with serverless code", "doc_url": "https://docs.microsoft.com/azure/azure-functions/"},
            {"name": "Container Instances", "description": "Easily run containers on Azure without managing servers", "doc_url": "https://docs.microsoft.com/azure/container-instances/"}
        ],
        "Storage": [
            {"name": "Blob Storage", "description": "REST-based object storage for unstructured data", "doc_url": "https://docs.microsoft.com/azure/storage/blobs/"},
            {"name": "Disk Storage", "description": "Persistent, secured disk options for Azure VMs", "doc_url": "https://docs.microsoft.com/azure/virtual-machines/disks-types"},
            {"name": "File Storage", "description": "File shares that use the standard SMB protocol", "doc_url": "https://docs.microsoft.com/azure/storage/files/"}
        ],
        "Databases": [
            {"name": "Azure SQL Database", "description": "Managed, intelligent SQL in the cloud", "doc_url": "https://docs.microsoft.com/azure/azure-sql/database/"},
            {"name": "Azure Cosmos DB", "description": "Globally distributed, multi-model database", "doc_url": "https://docs.microsoft.com/azure/cosmos-db/"}
        ],
        "AI + Machine Learning": [
            {"name": "Azure OpenAI Service", "description": "Apply advanced language models to variety of use cases", "doc_url": "https://docs.microsoft.com/azure/cognitive-services/openai/"},
            {"name": "Azure Machine Learning", "description": "Build, train, and deploy machine learning models", "doc_url": "https://docs.microsoft.com/azure/machine-learning/"},
            {"name": "Cognitive Services", "description": "Add cognitive capabilities to apps with APIs", "doc_url": "https://docs.microsoft.com/azure/cognitive-services/"}
        ],
        "Integration": [
            {"name": "Logic Apps", "description": "Automate the access and use of data across clouds", "doc_url": "https://docs.microsoft.com/azure/logic-apps/"},
            {"name": "API Management", "description": "Publish APIs to developers, partners, and employees securely", "doc_url": "https://docs.microsoft.com/azure/api-management/"}
        ]
    }
    
    # Common Azure concepts with explanations
    azure_concepts = {
        "Resource Group": "A container that holds related resources for an Azure solution. A resource group includes resources that you want to manage as a group.",
        "Azure Resource Manager": "Deployment and management service for Azure. Provides a management layer that enables you to create, update, and delete resources in your Azure account.",
        "Azure Portal": "Web-based, unified console that provides an alternative to command-line tools. You can manage your Azure subscription using the Azure portal.",
        "Azure CLI": "Command-line tool designed to get you working quickly with Azure, with an emphasis on automation.",
        "Azure PowerShell": "A set of cmdlets for managing Azure resources directly from the PowerShell command line.",
        "Subscription": "An agreement with Microsoft to use Azure services, and how you're billed for Azure services.",
        "Availability Set": "A logical grouping of VMs that allows Azure to understand how your application is built to provide redundancy and availability.",
        "Availability Zone": "Physically separate locations within an Azure region protected from data center failures."
    }
    
    # Common tasks and their step-by-step guides
    common_tasks = {
        "deploy virtual machine": "1. Sign in to the Azure portal\n2. Select 'Create a resource'\n3. Search for and select 'Virtual Machine'\n4. Fill in the basic details (name, region, image, size)\n5. Configure networking, management, and advanced options\n6. Review and create",
        "create storage account": "1. Sign in to the Azure portal\n2. Select 'Create a resource'\n3. Search for and select 'Storage account'\n4. Fill in the basics (subscription, resource group, name, region)\n5. Configure performance, redundancy, and advanced options\n6. Review and create",
        "set up azure functions": "1. Sign in to the Azure portal\n2. Select 'Create a resource'\n3. Search for and select 'Function App'\n4. Fill in the basics (subscription, resource group, name)\n5. Select runtime stack and version\n6. Choose region and hosting options\n7. Review and create"
    }

    @kernel_function(
        name="get_services_by_category",
        description="Gets a list of Azure services by category",
    )
    def get_services_by_category(
        self,
        category: str = "",
    ) -> str:
        """
        Gets a list of Azure services by category.
        If no category is provided, returns all categories.
        """
        if not category:
            # Return list of all categories
            return {
                "categories": list(self.azure_services.keys())
            }
        
        # Return services in the specified category (case-insensitive search)
        for cat, services in self.azure_services.items():
            if cat.lower() == category.lower():
                return {
                    "category": cat,
                    "services": services
                }
        
        # If category not found
        return {"error": f"Category '{category}' not found. Available categories: {', '.join(self.azure_services.keys())}"}

    @kernel_function(
        name="get_service_info",
        description="Gets detailed information about a specific Azure service",
    )
    def get_service_info(
        self,
        service_name: str,
    ) -> str:
        """
        Gets detailed information about a specific Azure service.
        """
        for category, services in self.azure_services.items():
            for service in services:
                if service["name"].lower() == service_name.lower():
                    result = service.copy()
                    result["category"] = category
                    return result
        
        # If service not found
        return {"error": f"Service '{service_name}' not found."}

    @kernel_function(
        name="explain_concept",
        description="Explains an Azure concept or terminology",
    )
    def explain_concept(
        self,
        concept: str,
    ) -> str:
        """
        Explains an Azure concept or terminology.
        """
        # Case-insensitive search
        for key, description in self.azure_concepts.items():
            if key.lower() == concept.lower():
                return {
                    "concept": key,
                    "explanation": description
                }
        
        # If concept not found
        return {"error": f"Concept '{concept}' not found in the knowledge base."}

    @kernel_function(
        name="guide_task",
        description="Provides step-by-step guidance for common Azure tasks",
    )
    def guide_task(
        self,
        task: str,
    ) -> str:
        """
        Provides step-by-step guidance for common Azure tasks.
        """
        # Find the closest matching task (simple contains check)
        for key, steps in self.common_tasks.items():
            if key.lower() in task.lower() or task.lower() in key.lower():
                return {
                    "task": key,
                    "steps": steps
                }
        
        # If task not found
        return {"error": f"No guidance found for task '{task}'. Try asking about deploying VMs, creating storage accounts, or setting up Azure Functions."}

    @kernel_function(
        name="search_documentation",
        description="Searches for Azure documentation on a given topic",
    )
    def search_documentation(
        self,
        query: str,
    ) -> str:
        """
        Returns relevant documentation links for Azure services based on the query.
        """
        results = []
        
        # Simple keyword matching for demo purposes
        query_lower = query.lower()
        for category, services in self.azure_services.items():
            for service in services:
                if (query_lower in service["name"].lower() or 
                    query_lower in service["description"].lower() or
                    query_lower in category.lower()):
                    results.append({
                        "service": service["name"],
                        "category": category,
                        "description": service["description"],
                        "documentation": service["doc_url"]
                    })
        
        if not results:
            return {"error": f"No documentation found for query '{query}'."}
            
        return {"results": results}

async def main():
    # Load environment variables
    load_dotenv()
    # Initialize the kernel
    kernel = Kernel()
    # Get configuration from environment variables
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
    print(f"Deployment: {deployment_name}")
    print(f"Endpoint: {base_url}")
    print(f"API Key: {api_key[:5]}...") # Only print first 5 chars for security
    # Add Azure OpenAI chat completion
    if not all([deployment_name, api_key, base_url]):
        raise ValueError("Missing required environment variables. Please check your .env file.")
    # Add Azure OpenAI chat completion
    chat_completion = AzureChatCompletion(
        deployment_name=deployment_name,
        api_key=api_key,
        base_url=base_url,
    )
    kernel.add_service(chat_completion)
    # Set the logging level for  semantic_kernel.kernel to DEBUG.
    setup_logging()
    logging.getLogger("kernel").setLevel(logging.DEBUG)
    
    # Add the AzureDocsPlugin
    kernel.add_plugin(
        AzureDocsPlugin(),
        plugin_name="AzureDocs",
    )
    
    # Enable planning
    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    
    # Create a history of the conversation
    history = ChatHistory()
    
    print("Azure Documentation Assistant initialized. Ask questions about Azure services, concepts, or tasks!")
    print("Type 'exit' to quit.")
    
    # Initiate a back-and-forth chat
    userInput = None
    while True:
        # Collect user input
        userInput = input("User > ")
        # Terminate the loop if the user says "exit"
        if userInput == "exit":
            break
        # Add user input to the history
        history.add_user_message(userInput)
        # Get the response from the AI
        result = await chat_completion.get_chat_message_content(
            chat_history=history,
            settings=execution_settings,
            kernel=kernel,
        )
        # Print the results
        print("Assistant > " + str(result))
        # Add the message from the agent to the chat history
        history.add_message(result)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
