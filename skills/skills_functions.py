"""Python implementations of the callable tools and their schemas."""
# Encapsulate the functions implemented in `memory.py` as "tools" for LLM to call

from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from skills.skills_memory import SkillMemory


@dataclass
class Parameter:
    """Represents a function parameter with its type and requirements."""
    name: str
    type: str
    description: str
    required: bool = True
    enum: Optional[List[str]] = None # Optional list of values (limiting the parameter values)


class ToolFunction: # Provide unified interface for defining tool functions and convert the function definition to OpenAI tool Schema
    """Base class for defining tool functions in a human-friendly way."""
    
    name: str
    description: str
    parameters: List[Parameter]
    
    @classmethod
    def execute(cls, memory: SkillMemory, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the function with the given arguments."""
        raise NotImplementedError("Subclasses must implement execute()")
    
    @classmethod
    def to_schema(cls, memory: SkillMemory = None) -> Dict[str, Any]:
        """Convert the function definition to OpenAI tool schema format.
        
        Args:
            memory: Memory instance to get configuration from
        """
        properties = {}
        required = []
        
        for param in cls.parameters:
            param_schema = {
                "type": param.type,
                "description": param.description
            }
            
            # Handle dynamic enum generation based on memory configuration
            if param.enum:
                param_schema["enum"] = param.enum

            properties[param.name] = param_schema
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": cls.name, # e.g., "new_memory_insert"
                "description": cls.description, # e.g., "Infer a new memory and append it to a memory store."
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# Example function definitions
class NewSkillInsert(ToolFunction):
    name = "new_skill_insert"
    description = "If there is no existing relevant skill, create new skill with desired skill name and content."
    parameters = [
        Parameter(
            name="skill_name",
            type="string",
            description="The name of the new skill to create."
        ),
        Parameter(
            name="content",
            type="string",
            description="The markdown content for the new skill."
        ),
    ]

    @classmethod
    def execute(cls, memory: SkillMemory, arguments: Dict[str, Any]) -> Dict[str, Any]:
        # 如果文件已存在，memory.new_memory_insert 会直接 raise FileExistsError
        # 异常会向上传递给 Agent 处理
        title = memory.new_memory_insert(arguments["skill_name"], arguments["content"])
        return {"status": "ok", "message": f"Skill created successfully.", "skill_name": title}


class SkillMemoryUpdate(ToolFunction):
    name = "skill_update"
    description = "If the existing skill can be improved, update the specific skill by its <skill_name>."
    parameters = [
        Parameter(
            name="skill_name",
            type="string",
            description="The name of the skill to update. Skill name must exist and exactly match the title of an existing skill."
        ),
        Parameter(
            name="new_name",
            type="string",
            description="The new skill name for the skill, which replaces the old name. If not provided, the skill name will remain unchanged.",
            required=False
        ),
        Parameter(
            name="new_content",
            type="string",
            description="The new content for the skill, which will replace the entire old content. Please ensure full content if provided. If not provided, the skill content will remain unchanged.",
            required=False
        ),
    ]

    @classmethod
    def execute(cls, memory: SkillMemory, arguments: Dict[str, Any]) -> Dict[str, Any]:
        updated_skill = memory.memory_update(
            title=arguments["skill_name"],
            new_name=arguments.get("new_name"),
            new_content=arguments.get("new_content")
        )
        return {"status": "ok", "message": f"Skill '{arguments['skill_name']}' updated successfully.", "updated_skill": updated_skill}


class SkillDelete(ToolFunction):
    name = "skill_delete"
    description = "Delete an existing skill by its title."
    parameters = [
        Parameter(
            name="skill_name",
            type="string",
            description="The name of the skill to delete."
        ),
    ]

    @classmethod
    def execute(cls, memory: SkillMemory, arguments: Dict[str, Any]) -> Dict[str, Any]:
        memory.memory_delete(arguments["skill_name"])
        return {"status": "ok", "message": f"Skill '{arguments['skill_name']}' deleted successfully."}


class SearchSkill(ToolFunction):
    name = "search_skill"
    description = "Search for skills using BM25 or text embedding similarity."
    parameters = [
        Parameter(
            name="query",
            type="string",
            description="Query string to search for in skill title and content."
        ),
        Parameter(
            name="search_method",
            type="string",
            description="Search method to use: 'bm25' for keyword-based search or 'text-embedding' for semantic similarity search",
            required=False,
            enum=["bm25", "text-embedding"]
        )
    ]
    
    @classmethod
    def execute(cls, memory: SkillMemory, arguments: Dict[str, Any]) -> Dict[str, Any]:
        query = arguments["query"]
        search_method = arguments.get("search_method", "bm25")
        
        results = memory.memory_search(
            query=query,
            top_k=memory.TOPK,
            search_method=search_method
        )
        return {"status": "ok", "results": results, "search_method": search_method}


# List of all available tool functions
SKILL_MEMORY_TOOL_FUNCTIONS = [
    NewSkillInsert,
    SkillMemoryUpdate,
    SkillDelete,
]

SEARCH_TOOL_FUNCTIONS = [
    SearchSkill,
]


# Generate the function implementations map
FUNCTION_IMPLS = { # Map classmethod to tool names
    func.name: func.execute for func in SKILL_MEMORY_TOOL_FUNCTIONS + SEARCH_TOOL_FUNCTIONS
}

# Generate the OpenAI tool schemas - these functions now require a memory instance
def get_memory_tool_schemas(memory: SkillMemory) -> List[Dict[str, Any]]:
    """Generate OpenAI tool schemas for memory functions based on memory configuration."""
    return [func.to_schema(memory) for func in SKILL_MEMORY_TOOL_FUNCTIONS]

def get_search_tool_schemas(memory: SkillMemory) -> List[Dict[str, Any]]:
    """Generate OpenAI tool schemas for search functions based on memory configuration."""
    return [func.to_schema(memory) for func in SEARCH_TOOL_FUNCTIONS]

# Backward compatibility - these will work but won't respect including_core configuration
MEMORY_TOOL_SCHEMAS = [ # Map classmethod to OpenAI tool schemas
    func.to_schema() for func in SKILL_MEMORY_TOOL_FUNCTIONS
]

SEARCH_TOOL_SCHEMAS = [ # Map classmethod to OpenAI tool schemas
    func.to_schema() for func in SEARCH_TOOL_FUNCTIONS
]