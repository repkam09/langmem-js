import { BaseMessage } from "@langchain/core/messages";

function getMsgTitleRepr(title: string): string {
  const padded = ` ${title} `;
  const sepLen = Math.floor((80 - padded.length) / 2);
  const sep = "=".repeat(sepLen);
  const secondSep = padded.length % 2 === 0 ? sep : sep + "=";
  return `${sep}${padded}${secondSep}`;
}

export function getTrajectoryClean(
  messages: (BaseMessage | Record<string, unknown>)[]
): string {
  const response: string[] = [];
  for (const m of messages) {
    if (m instanceof BaseMessage) {
      const type = m._getType();
      const roleMap: Record<string, string> = {
        human: "Human",
        ai: "AI",
        system: "System",
        tool: "Tool",
      };
      const role = roleMap[type] ?? type;
      const title = getMsgTitleRepr(role);
      const content =
        typeof m.content === "string" ? m.content : JSON.stringify(m.content);
      response.push(`${title}\n\n${content}`);
    } else if (
      m &&
      typeof m === "object" &&
      "role" in m &&
      "content" in m
    ) {
      const role = String((m as any).role);
      const title = getMsgTitleRepr(role);
      const name = (m as any).name;
      let header = title;
      if (name !== undefined) {
        header += `\nName: ${name}`;
      }
      response.push(`${header}\n\n${String((m as any).content)}`);
    }
  }
  return response.join("\n");
}
