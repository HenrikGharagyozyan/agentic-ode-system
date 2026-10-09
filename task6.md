Task 6. Agentic System
Task 5 asks for the differential equation, a plan for the network, and a script that has to be written
and run. This task builds a system of four roles that splits that work, and then asks the system to
do Task 5.
Problem.
The user speaks only to the orchestrator. Every other role speaks only when an edge of the graph
gives it the turn. The dialogue is the state of the graph. Each turn carries the name of the role that
produced it and the text of what that role said.
The graph. Build the system with LangGraph, as a StateGraph. The orchestrator, the scientist,
the general agent, and the executor are four nodes. The state they share is the dialogue. An edge
from one node to another hands that agent the turn: the node reads the messages already in the
state, including what the other agents wrote, and appends its own message. The orchestrator node
chooses the next edge, to the scientist, the general agent, the executor, or the end. After a specialist
node runs, an edge returns the turn, so the next speaker sees the new message. That is how the
agents communicate: each one writes into the shared state, and the next one reads it.
Orchestrator. On each turn the orchestrator sees the dialogue and returns a decision with three
fields: the next role, the instruction that role should follow, and a one-sentence reason. The next
role is one of scientist, general, executor, or finish. That decision is itself a call to a model,
which reads the dialogue and chooses. A table of keywords, or a fixed script that always calls the
same roles in the same order, does not count. The orchestrator writes no theory and no code, and
it has no coding tool. It stops on finish, or after 8 decisions, whichever comes first. It may finish
only after a turn from the executor contains a test RMSE printed by a script that actually ran.
Scientist. This role owns the theory and the plan. From the problem it writes the differential
equation, the network it proposes, the state equation to integrate, how to train, the free-run test,
and a numbered list of steps that is specific enough to code. It has no coding tool. It does not claim
that a number was measured unless an executor turn already printed that number.
General agent. This role answers a high-level question that sits beside the fit: what a record in
the file is, what a term in the problem means, what a reported error is measuring. It has no coding
tool, and it does not write the fitting plan. When a question is the plan for the differential equation,
the orchestrator sends it to the scientist. When a question is a traceback, the orchestrator sends it
to the executor.
Executor and the coding tool. The executor is the only role that receives the coding tool. The
tool does two things, and both are required.
• Write a Python file into a working directory chosen for this run.
6
• Run that file, with a time limit of five minutes, and return the exit code, the standard output,
and the standard error.
The executor implements the latest plan from the scientist, in a new file. It returns the free-run
RMSE the script printed. When the script fails, it returns the traceback. A later executor turn may
repair the script. The scientific plan stays the one the scientist wrote: the same differential equation
and the same free-run test.
Validate the system on Task 5. Give the system Task 5 and let it try to solve that task. In the
solution PDF, show the process and the result: which agent did what, and the free-run error the run
printed. Push the code to a GitHub repository and put the link in the PDF. 14 points