VkResult vk_replay_metadata_allocate(size_t cost, int clear, RcBudgetProducer producer,
    RcBudgetTicket *ticket, void **allocation) {
    assert(producer == RC_BUDGET_SNAPSHOT && !clear && cost && !ticket->token.ledger);
    const RcAdmissionVector demand = {.amount = {[RC_BUDGET_RAM] = cost, [RC_BUDGET_COMMIT] = cost}};
    *allocation = NULL;
    if (rc_admission_reserve(&budget, producer, &demand, &ticket->token) != RC_ADMISSION_OK)
        return VK_NOT_READY;
    ticket->producer = producer;
    *allocation = malloc(cost);
    assert(*allocation && resource_core_budget_materialize(ticket, &demand) == RC_ADMISSION_OK);
    return VK_SUCCESS;
}
void vk_replay_metadata_free(void *allocation, RcBudgetTicket *ticket) {
    assert(ticket->producer == RC_BUDGET_SNAPSHOT && resource_core_budget_retire(ticket) == RC_ADMISSION_OK);
    free(allocation);
    assert(resource_core_budget_destroyed(ticket) == RC_ADMISSION_OK);
}
