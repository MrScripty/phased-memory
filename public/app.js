'use strict';
const $=id=>document.getElementById(id);
const token=document.querySelector('meta[name="lab-session"]').content;
const claim={id:'claim-1',normalized:'Alice believes Bob is in Paris.',triple:{subject:'bob',predicate:'location',object:'paris',polarity:'affirmed'},entities:['alice','bob','paris'],source:'Alice statement',perspective:'alice',continuity:'story-main',story_time:null,qualifiers:['belief']};
$('batch').value=JSON.stringify({raw_text:$('raw').value,claims:[claim],provenance:{kind:'human',note:'human-authored structured claim'},embeddings:[]},null,2);
function parsed(id){return JSON.parse($(id).value);}
async function request(body){
  document.querySelectorAll('button').forEach(b=>b.disabled=true);$('status').textContent='Operation running. Generation has no imposed timeout. No automatic retry.';
  try{const response=await fetch('/api',{method:'POST',headers:{'Content-Type':'application/json','X-Lab-Session':token},body:JSON.stringify(body)});const value=await response.json();$('response').textContent=JSON.stringify(value,null,2);if(!response.ok)throw new Error(value.error||`HTTP ${response.status}`);$('status').textContent=value.status||'Response observed. Inspect provenance and result.';return value;}
  catch(error){$('status').textContent=error.message;throw error;}
  finally{document.querySelectorAll('button').forEach(b=>b.disabled=false);}
}
function bind(id,callback){$(id).addEventListener('click',async()=>{try{await callback();}catch(error){$('status').textContent=error.message;}});}
bind('discover',async()=>{$('runtime').textContent=JSON.stringify(await request({action:'discover'}),null,2);});
bind('extract',async()=>{const v=await request({action:'extract',raw_text:$('raw').value});$('status').textContent='Extraction preview returned; model provenance is in the latest response.';return v;});
bind('ingest-raw',()=>request({action:'ingest_raw',raw_text:$('raw').value}));
bind('ingest',()=>request({action:'ingest',batch:parsed('batch'),embed:$('embed').checked}));
bind('save-relation',()=>request({action:'relation',relation:parsed('relation')}));
function render(id,hits){const target=$(id);target.replaceChildren();if(hits===null){target.textContent='Unavailable. See the relationship status.';return;}if(!hits.length){target.textContent='No matching memories in this embedding space.';return;}for(const hit of hits){const card=document.createElement('div');card.className='hit';const title=document.createElement('strong');title.textContent=hit.claim.id+' · '+hit.claim.normalized;const detail=document.createElement('pre');detail.textContent=JSON.stringify({cosine:hit.score,triple:hit.claim.triple,perspective:hit.claim.perspective,continuity:hit.claim.continuity,story_time:hit.claim.story_time,qualifiers:hit.claim.qualifiers,embedding_provenance:hit.embedding_provenance,decision:hit.decision},null,2);card.append(title,detail);target.append(card);}}
bind('search',async()=>{const v=await request({action:'query',text:$('query').value,embedding:$('synthetic').checked?parsed('query-embedding'):null,filters:parsed('filters'),rerank:$('rerank').checked});for(const id of ['vector','structured','reranked'])render(id,v[id]);$('rerank-status').textContent=v.rerank_status;$('relations').textContent=JSON.stringify(v.relations,null,2);});
